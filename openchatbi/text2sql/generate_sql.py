import datetime
import re
import time as _time
from collections.abc import Callable
from typing import Any

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from openchatbi import config
from openchatbi.catalog import CatalogStore
from openchatbi.constants import (
    SQL_EXECUTE_TIMEOUT,
    SQL_NA,
    SQL_RESULT_LIMIT,
    SQL_SECURITY_ERROR,
    SQL_SUCCESS,
    datetime_format,
)
from openchatbi.graph_state import SQLGraphState
from openchatbi.memory_config import get_memory_config
from openchatbi.memory_scoring import composite_score
from openchatbi.observability.audit import AuditLogger
from openchatbi.observability.context import get_run_context
from openchatbi.prompts.system_prompt import get_text2sql_dialect_prompt_template
from openchatbi.text2sql.confidence import SimpleSQLEvaluator
from openchatbi.text2sql.data import get_learned_sql_store, sql_example_dicts, sql_example_retriever
from openchatbi.text2sql.errors import (
    EmptyResultError,
    SQLSecurityError,
    Text2SQLError,
    classify_sql_exception,
)
from openchatbi.text2sql.visualization import VisualizationService
from openchatbi.utils import get_text_from_content, log

_audit_logger = AuditLogger()


_COLUMN_PROMPT_TEMPLATE = """### Columns
Column(Name, Type, Display Name, Description):
[
{}
]
"""


def _limit_sql_query(sql: str, limit: int = SQL_RESULT_LIMIT) -> str:
    """Wrap a query so Text2SQL never executes an unbounded result set."""
    normalized_sql = sql.strip().rstrip(";").strip()
    # Generated SQL is validated separately; this wrapper only adds a result limit.
    return f"SELECT * FROM (\n{normalized_sql}\n) AS openchatbi_limited_result LIMIT {limit}"  # nosec


def _validate_sql_safety(sql: str) -> tuple[bool, str]:
    """Validate generated SQL before execution."""
    disallowed_patterns = [
        r"(?:^|\s)INSERT\s+(?:INTO\s+|OVERWRITE\s+)(?:TABLE\s+)?",
        r"(?:^|\s)UPDATE\s+[`\"\[\w]",
        r"(?:^|\s)DELETE\s+(?:FROM\b|[`\"\[\w]+\s+FROM\b)",
        r"(?:^|\s)DROP\s+(?:TEMP(?:ORARY)?\s+)?(?:TABLE|DATABASE|SCHEMA|VIEW|MATERIALIZED\s+VIEW|USER|ROLE|INDEX|FUNCTION|PROCEDURE|TRIGGER)\b",
        r"(?:^|\s)CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?(?:TABLE|DATABASE|SCHEMA|VIEW|MATERIALIZED\s+VIEW|USER|ROLE|INDEX|FUNCTION|PROCEDURE|TRIGGER)\b",
        r"(?:^|\s)ALTER\s+(?:TABLE|DATABASE|SCHEMA|VIEW|MATERIALIZED\s+VIEW|USER|ROLE|INDEX|FUNCTION|PROCEDURE|TRIGGER)\b",
        r"(?:^|\s)TRUNCATE\b",
        r"(?:^|\s)GRANT\b",
        r"(?:^|\s)REVOKE\b",
        r"(?:^|\s)LOAD\s+DATA\b",
        r"(?:^|\s)INTO\s+(?:OUT|DUMP)FILE\b",
        r"(?:^|\s)FOR\s+UPDATE\b",
    ]
    for pattern in disallowed_patterns:
        if re.search(pattern, sql, flags=re.IGNORECASE):
            return False, f"Operation not allowed: {pattern}"

    return True, ""


def _get_sql_result_limit_config() -> tuple[bool, int]:
    """Read SQL result limit settings, defaulting to safe bounded execution."""
    try:
        cfg = config.get()
    except ValueError:
        return True, SQL_RESULT_LIMIT

    enabled = getattr(cfg, "enable_sql_result_limit", True)
    limit = getattr(cfg, "sql_result_limit", SQL_RESULT_LIMIT)
    if not isinstance(limit, int) or limit <= 0:
        limit = SQL_RESULT_LIMIT
    return bool(enabled), limit


def _get_empty_result_config() -> tuple[bool, None]:
    """Read whether zero-row results should be treated as a soft failure.

    Defaults to OFF so empty results stay SQL_SUCCESS (preserves the existing
    generate_visualization entry path). Returns a tuple for symmetry with the
    result-limit config reader.
    """
    try:
        cfg = config.get()
    except ValueError:
        return False, None
    return bool(getattr(cfg, "enable_empty_result_error", False)), None


def _extract_exception_message(exc: BaseException) -> str:
    """Collect exception text from SQLAlchemy wrappers and native DB errors."""
    message_parts = [str(exc)]
    orig_error = getattr(exc, "orig", None)
    if orig_error is not None:
        message_parts.append(str(orig_error))
        orig_args = getattr(orig_error, "args", None) or []
        message_parts.extend(str(arg) for arg in orig_args if arg is not None)

    exc_args = getattr(exc, "args", None) or []
    message_parts.extend(str(arg) for arg in exc_args if arg is not None)
    return " ".join(message_parts).lower()


def _classify_operational_error(exc: OperationalError) -> str:
    """Classify operational errors into timeout/connection, syntax, or other."""
    timeout_or_connection_markers = (
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "connection aborted",
        "connection closed",
        "connection failed",
        "server has gone away",
        "lost connection",
        "could not connect",
        "can't connect",
        "network is unreachable",
    )
    syntax_error_markers = (
        "syntax error",
        "parse error",
        "unexpected token",
        "unexpected character",
        "unexpected end of input",
        "invalid syntax",
        "near ",
    )

    message = _extract_exception_message(exc)
    if getattr(exc, "connection_invalidated", False):
        return "timeout_or_connection"
    if any(marker in message for marker in timeout_or_connection_markers):
        return "timeout_or_connection"
    if any(marker in message for marker in syntax_error_markers):
        return "syntax"
    return "other"


def create_sql_nodes(
    llm: BaseChatModel,
    catalog: CatalogStore,
    dialect: str,
    visualization_mode: str | None = "rule",
    learned_sql_store: Any | None = None,
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Creates the SQL processing nodes for LangGraph.

    Args:
        llm (BaseChatModel): The language model to use for SQL generation.
        catalog (CatalogStore): The catalog store containing schema information.
        dialect (str): The SQL dialect to use (e.g., 'presto', 'mysql').
        visualization_mode (str | None): Visualization analysis mode ("rule", "llm", or None to skip).
        learned_sql_store: Optional LearnedSQLStore handle for blended retrieval and
            gated auto-capture. When None, falls back to the legacy static retriever
            and capture is a no-op (default-off, zero regression).

    Returns:
        tuple: Six node functions (generate, execute, regenerate, visualization, score_sql, confidence_gate)
    """

    # Initialize visualization service based on configuration
    visualization_service = VisualizationService(llm if visualization_mode == "llm" else None)

    def _get_column_prompt(column: dict[str, Any]) -> str:
        alias_prompt = f"alias({column['alias']})" if "alias" in column and column["alias"] else ""
        return (
            f"""    Column("{column['column_name']}", {column['type']}, {column['display_name']},"""
            f""" "{alias_prompt}{column['description']}"),"""
        )

    def _get_table_schema_prompt(tables_columns: list[dict[str, Any]]) -> str:
        """Generates a prompt string for table schemas, including table description,
        columns, derived metrics and rules when writting SQL

        Args:
            tables_columns (List[Dict[str, Any]]): List of tables with selected columns.

        Returns:
            str: Formatted table schema prompt string.
        """
        schema_prompt = []
        for table_dict in tables_columns:
            table_name = table_dict["table"]
            # TODO maybe use columns in prompt
            columns = table_dict["columns"]
            table_info = catalog.get_table_information(table_name)
            single_table_schema_prompt = f"## Table {table_name}\n{table_info['description']}\n"
            columns = catalog.get_column_list(table_name)
            single_table_schema_prompt += _COLUMN_PROMPT_TEMPLATE.format(
                "\n".join([_get_column_prompt(column) for column in columns])
            )
            single_table_schema_prompt += table_info.get("derived_metric", "")
            single_table_schema_prompt += table_info["sql_rule"]
            schema_prompt.append(single_table_schema_prompt)
        return "\n".join(schema_prompt)

    def _get_relevant_sql_examples_prompt(question, tables_columns: list[dict[str, Any]]) -> str:
        """Retrieves relevant SQL examples based on the question and selected tables.

        Blends static + golden + auto-captured patterns via LearnedSQLStore when a
        store is wired and pattern memory is enabled; otherwise preserves the legacy
        static-retriever behavior (strict subset filter) for zero regression.

        Args:
            question (str): The natural language question.
            tables_columns (List[str]): List of selected tables with selected columns.

        Returns:
            str: Formatted string of relevant SQL examples.
        """
        tables = [d["table"] for d in tables_columns]
        mem_cfg = get_memory_config()

        if learned_sql_store is not None and getattr(mem_cfg, "enable_pattern_memory", False):
            cap = int(getattr(mem_cfg, "max_patterns_per_query", 5))

            # Convention #5: score_fn is a (metadata, base_rank) adapter wrapping composite_score.
            def score_fn(meta: dict, base_rank: int) -> float:
                return composite_score(
                    1.0 / (1.0 + base_rank),
                    float(meta.get("importance", 1.0) or 1.0),
                    meta.get("last_used", ""),
                    int(meta.get("use_count", 0) or 0),
                    mem_cfg,
                )

            try:
                retrieved = learned_sql_store.retrieve(question, k=max(cap * 2, 10), score_fn=score_fn)
            except Exception as e:  # fall back to static path on store failure
                log(f"Blended retrieval failed, falling back to static examples: {e}")
                retrieved = None
            if retrieved is not None:
                examples = []
                for ex_question, example_sql, used_tables in retrieved:
                    # Soft filter: keep when the pattern touches any selected table
                    # (relaxed from strict subset so learned patterns aren't silently dropped).
                    if not used_tables or any(t in tables for t in used_tables):
                        examples.append(f"<example>\nQ: {ex_question}\nA: {example_sql}\n</example>\n")
                    if len(examples) >= cap:
                        break
                if examples:
                    log(f"Blended examples (store): {examples}")
                    return "\n".join(examples)
                # Everything was filtered out (or the store is empty): fall through
                # to the legacy static path instead of returning no examples at all.

        # Legacy path: static retriever + strict subset filter (unchanged behavior).
        relevant_questions = sql_example_retriever.invoke(question)
        # log(f"Retrieved examples for question: {question} \n Relevant questions: {relevant_questions}")
        # filter examples that only use the selected tables
        examples = []
        for relevant_document in relevant_questions:
            question = relevant_document.page_content
            example_sql, used_tables = sql_example_dicts[question]
            if all(table in tables for table in used_tables):
                examples.append(f"<example>\nQ: {question}\nA: {example_sql}\n</example>\n")
        log(f"Examples using selected tables: {examples}")
        return "\n".join(examples)

    def _analyze_dataframe_schema(df: pd.DataFrame) -> dict[str, Any]:
        """Analyze DataFrame to understand column types and characteristics."""
        try:
            schema_info: dict[str, Any] = {
                "columns": list(df.columns),
                "column_types": {},
                "row_count": len(df),
                "numeric_columns": [],
                "categorical_columns": [],
                "datetime_columns": [],
            }

            for col in df.columns:
                dtype = str(df[col].dtype)
                schema_info["column_types"][col] = dtype

                # Classify column types
                if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                    schema_info["numeric_columns"].append(col)
                elif df[col].dtype == "object":
                    # Check if it could be datetime
                    try:
                        pd.to_datetime(df[col].head(10))
                        schema_info["datetime_columns"].append(col)
                    except Exception:
                        schema_info["categorical_columns"].append(col)

            # Calculate unique value counts for categorical columns
            schema_info["unique_counts"] = {}
            for col in schema_info["categorical_columns"]:
                schema_info["unique_counts"][col] = df[col].nunique()

            return schema_info
        except Exception as e:
            return {"error": f"Failed to analyze data schema: {str(e)}"}

    def _execute_sql(sql: str) -> tuple[dict, str]:
        """Executes the generated SQL query and returns the result with schema analysis.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            Tuple[dict, str]: A tuple containing (schema_info, CSV string).
        """
        limit_enabled, result_limit = _get_sql_result_limit_config()
        is_safe, reason = _validate_sql_safety(sql)
        if not is_safe:
            raise SQLSecurityError(reason)

        if limit_enabled:
            execute_sql = _limit_sql_query(sql, result_limit)
        else:
            execute_sql = sql
        with catalog.get_sql_engine().connect() as connection:
            result = connection.execute(text(execute_sql))

            # Keep the agent context bounded even if a backend ignores or rewrites the LIMIT.
            rows = result.fetchmany(result_limit) if limit_enabled else result.fetchall()

            # Get column names
            columns = list(result.keys())

            # Create DataFrame for analysis
            df = pd.DataFrame(rows, columns=columns)

            # Analyze data schema
            schema_info = _analyze_dataframe_schema(df)
            if limit_enabled:
                schema_info["result_limit"] = result_limit

            # Format as CSV
            csv_data = df.to_csv(index=False)

            connection.commit()
            return schema_info, csv_data

    def _maybe_capture_pattern(state: SQLGraphState, human_approved: bool = False) -> None:
        """Capture (question -> SQL -> tables) into LearnedSQLStore on approval.

        Called from confidence_gate_node's approve paths only, so approval
        semantics (Convention #10) are guaranteed structurally. Reads the
        pre-computed sql_confidence from state (single LLM evaluation in
        score_sql_node - no second evaluator call here). A human approval
        overrides the score threshold. Never raises.
        """
        if learned_sql_store is None:
            return
        try:
            mem_cfg = get_memory_config()
            if not getattr(mem_cfg, "enable_pattern_memory", False):
                return
            question = state.get("rewrite_question", "")
            sql_query = state.get("sql", "").strip()
            tables = [d["table"] for d in state.get("tables", []) if isinstance(d, dict) and d.get("table")]
            if not question or not sql_query:
                return
            # S2 gate (success != correct): only promote SQL scored above the
            # threshold, unless a human explicitly approved it.
            if not human_approved:
                score = state.get("sql_confidence")
                if score is None:
                    log("Pattern capture skipped: no confidence score available")
                    return
                try:
                    threshold = float(getattr(config.get(), "sql_confidence_threshold", 0.7))
                except ValueError:
                    threshold = 0.7
                if score < threshold:
                    log(f"Pattern capture skipped: confidence {score:.2f} < {threshold:.2f}")
                    return
            retry_count = int(state.get("sql_retry_count", 0) or 0)
            importance = 1.0 / (1.0 + retry_count)  # first-try success weighted highest
            learned_sql_store.add(
                question,
                sql_query,
                tables,
                source="auto",
                importance=importance,
                namespace=getattr(mem_cfg, "pattern_scope", "global"),
            )
        except Exception as e:  # never let capture break the response
            log(f"Pattern capture error (ignored): {e}")

    def generate_sql_node(state: SQLGraphState) -> dict:
        """First node: Generates initial SQL query based on the state.

        Args:
            state (SQLGraphState): The current SQL graph state containing the question and tables.

        Returns:
            dict: Updated state with generated SQL query.
        """
        if not state.get("rewrite_question"):
            log("Missing rewrite question, skipping SQL generation.")
            return {}
        if not state.get("tables"):
            log("Missing tables, skipping SQL generation.")
            return {}

        question = state["rewrite_question"]
        tables_columns = state["tables"]
        system_prompt = (
            get_text2sql_dialect_prompt_template(dialect)
            .replace("[table_schema]", _get_table_schema_prompt(tables_columns))
            .replace("[examples]", _get_relevant_sql_examples_prompt(question, tables_columns))
            .replace("[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format))
        )

        user_prompt = f"""Generate a SQL query for the question: {question}"""
        messages = [SystemMessage(system_prompt)] + list(state["messages"]) + [HumanMessage(user_prompt)]

        response = llm.invoke(messages)
        response_content = get_text_from_content(response.content)
        sql_query = response_content.replace("```sql", "").replace("```", "").strip()

        if not sql_query or sql_query.lower() == "null":
            log(f"Generated SQL query is empty. LLM output: {response.content}")
            return {
                "messages": [AIMessage(response_content)],
                "sql": sql_query,
                "sql_retry_count": 0,
                "sql_execution_result": "",
                "previous_sql_errors": [],
            }

        return {"sql": sql_query, "sql_retry_count": 0, "sql_execution_result": "", "previous_sql_errors": []}

    def execute_sql_node(state: SQLGraphState) -> dict:
        """Second node: Executes the SQL query and returns result or error.

        Args:
            state (SQLGraphState): The current SQL graph state containing the SQL query.

        Returns:
            dict: Updated state with execution result or error information.
        """
        sql_query = state.get("sql", "").strip()
        if not sql_query:
            return {"sql_execution_result": SQL_NA, "messages": [AIMessage("No SQL query to execute")]}

        user_id, _ = get_run_context()
        _start = _time.time()

        try:
            schema_info, csv_result = _execute_sql(sql_query)
            duration_ms = (_time.time() - _start) * 1000.0
            row_count = schema_info.get("row_count") if isinstance(schema_info, dict) else None

            # Empty-result gate (default OFF — zero rows stay SQL_SUCCESS).
            empty_result_enabled, _ = _get_empty_result_config()
            if empty_result_enabled and row_count == 0:
                _audit_logger.log_sql_exec(
                    sql=sql_query,
                    dialect=dialect,
                    row_count=0,
                    duration_ms=duration_ms,
                    status=SQL_NA,
                    user_id=user_id,
                )
                previous_errors = list(state.get("previous_sql_errors", []))
                attempt = len(previous_errors) + 1
                err: Text2SQLError = EmptyResultError("Query returned no rows")
                previous_errors.append(
                    {
                        "sql": sql_query,
                        "error": f"{err.error_type}: no rows returned",
                        "error_type": err.error_type,
                        "error_code": err.code,
                        "error_class": type(err).__name__,
                        "recovery_strategy": err.recovery_strategy.value,
                        "attempt": attempt,
                    }
                )
                return {
                    "sql_execution_result": SQL_NA,
                    "previous_sql_errors": previous_errors,
                }

            _audit_logger.log_sql_exec(
                sql=sql_query,
                dialect=dialect,
                row_count=row_count,
                duration_ms=duration_ms,
                status=SQL_SUCCESS,
                user_id=user_id,
            )
            if "result_limit" in schema_info:
                result_label = f"SQL Result (limited to first {schema_info['result_limit']} rows)"
            else:
                result_label = "SQL Result"
            result = f"```sql\n{sql_query}\n```\n{result_label}:\n```csv\n{csv_result}\n```"
            # Pattern auto-capture happens in confidence_gate_node (approve paths),
            # reusing the single score_sql_node evaluation - no LLM call here.
            return {
                "sql_execution_result": SQL_SUCCESS,
                "schema_info": schema_info,
                "data": csv_result,
                "messages": [AIMessage(result)],
            }
        except Exception as e:
            err = classify_sql_exception(e)
            log(f"{err.error_type}: {str(e)}")

            _audit_logger.log_sql_exec(
                sql=sql_query,
                dialect=dialect,
                row_count=None,
                duration_ms=(_time.time() - _start) * 1000.0,
                status=err.code,
                user_id=user_id,
                error=str(e),
            )

            previous_errors = list(state.get("previous_sql_errors", []))
            attempt = len(previous_errors) + 1
            previous_errors.append(
                {
                    "sql": sql_query,
                    "error": f"{err.error_type}: {str(e)}",
                    "error_type": err.error_type,
                    "error_code": err.code,
                    "error_class": type(err).__name__,
                    "recovery_strategy": err.recovery_strategy.value,
                    "attempt": attempt,
                }
            )

            update: dict = {
                "sql_execution_result": err.code,
                "previous_sql_errors": previous_errors,
            }
            # Branches that historically surfaced a message to the user keep doing so.
            if err.code == SQL_EXECUTE_TIMEOUT:
                error_result = (
                    f"```sql\n{sql_query}\n```\nDatabase Connection Timeout: {str(e)}\n"
                    "Please check database connectivity."
                )
                update["messages"] = [AIMessage(error_result)]
            elif err.code == SQL_SECURITY_ERROR:
                error_result = f"```sql\n{sql_query}\n```\n{err.error_type}: {str(e)}"
                update["messages"] = [AIMessage(error_result)]
            return update

    def regenerate_sql_node(state: SQLGraphState) -> dict:
        """Third node: Regenerates SQL based on previous errors.

        Args:
            state (SQLGraphState): The current SQL graph state containing error information.

        Returns:
            dict: Updated state with regenerated SQL query.
        """
        question = state["rewrite_question"]
        tables = state["tables"]
        previous_errors = state.get("previous_sql_errors", [])
        retry_count = state.get("sql_retry_count", 0) + 1

        system_prompt = (
            get_text2sql_dialect_prompt_template(dialect)
            .replace("[table_schema]", _get_table_schema_prompt(tables))
            .replace("[examples]", _get_relevant_sql_examples_prompt(question, tables))
            .replace("[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format))
        )

        user_prompt = f"""Generate a SQL query for the question: {question}"""
        if previous_errors:
            user_prompt += "\n\nPrevious attempts failed with errors:"
            for i, error_info in enumerate(previous_errors, 1):
                error_type_hint = error_info.get("error_type", "")
                hint_line = f"\nError type: {error_type_hint}" if error_type_hint else ""
                user_prompt += (
                    f"\n\nAttempt {i}:\nSQL: {error_info['sql']}" f"{hint_line}\nError: {error_info['error']}"
                )
            user_prompt += "\n\nPlease analyze the errors above and generate a corrected SQL query."

        messages = [SystemMessage(system_prompt)] + list(state["messages"]) + [HumanMessage(user_prompt)]

        response = llm.invoke(messages)
        response_content = get_text_from_content(response.content)
        sql_query = response_content.replace("```sql", "").replace("```", "").strip()

        last_strategy = previous_errors[-1].get("recovery_strategy", "") if previous_errors else ""
        if not sql_query:
            log(f"Generated SQL query is empty. LLM output: {response.content}")
            error_result = f"Failed to regenerate valid SQL after {retry_count} attempts."
            return {
                "messages": [AIMessage(error_result)],
                "sql": "",
                "sql_retry_count": retry_count,
                "sql_execution_result": SQL_NA,
                "recovery_strategy": last_strategy,
            }

        return {
            "sql": sql_query,
            "sql_retry_count": retry_count,
            "sql_execution_result": "",
            "recovery_strategy": last_strategy,
        }

    def generate_visualization_node(state: SQLGraphState) -> dict:
        """Fourth node: Generates visualization DSL based on successful SQL execution result.

        Args:
            state (SQLGraphState): The current SQL graph state containing query data and results.

        Returns:
            dict: Updated state with visualization DSL.
        """
        execution_result = state.get("sql_execution_result", "")
        if execution_result != SQL_SUCCESS:
            # No visualization for failed queries
            return {"visualization_dsl": {}}

        question = state.get("rewrite_question", "")
        schema_info = state.get("schema_info", {})
        data = state.get("data", "")

        if not question or not schema_info or not data or not visualization_mode:
            return {"visualization_dsl": {}}

        try:
            # Generate visualization DSL using configured service
            viz_dsl = visualization_service.generate_visualization(question, schema_info, data)

            # Handle case where visualization is skipped
            if viz_dsl is None:
                return {"visualization_dsl": {}}

            # Update the AI message to include visualization information
            messages = list(state.get("messages", []))
            if messages and hasattr(messages[-1], "content"):
                current_content = messages[-1].content
                viz_info = f"\n\n**Visualization Generated**: {viz_dsl.chart_type.title()} chart with {len(viz_dsl.data_columns)} column(s)"
                if isinstance(current_content, str):
                    messages[-1] = AIMessage(current_content + viz_info)

            return {"visualization_dsl": viz_dsl.to_dict(), "messages": messages}
        except Exception as e:
            log(f"Visualization generation error: {str(e)}")
            return {"visualization_dsl": {"error": f"Failed to generate visualization: {str(e)}"}}

    def score_sql_node(state: SQLGraphState) -> dict:
        """Score the executed SQL with the S2 confidence evaluator.

        Only runs after a successful execution (post_exec mode); other
        execution results are passed through unscored.

        Cost-inert by default: skips the LLM evaluator entirely when neither
        the confidence gate nor pattern memory is enabled.
        """
        if state.get("sql_execution_result", "") != SQL_SUCCESS:
            return {}
        sql_query = state.get("sql", "").strip()
        if not sql_query:
            return {}

        # Short-circuit: no LLM call when neither consumer is enabled.
        try:
            cfg = config.get()
            gate_on = getattr(cfg, "enable_confidence_gate", False)
        except ValueError:
            gate_on = False
        # enable_pattern_memory lives on memory_config (Task 13), NOT as a top-level
        # Config attr; read it via get_memory_config so the auto-capture confidence
        # gate actually gets a score when pattern memory is on.
        try:
            pattern_on = bool(getattr(get_memory_config(), "enable_pattern_memory", False))
        except Exception:
            pattern_on = False
        if not gate_on and not pattern_on:
            return {}  # cost-inert: no scoring needed when neither consumer is enabled

        question = state.get("rewrite_question", "")
        schema_info = state.get("schema_info", {})
        data_sample = state.get("data", "")
        try:
            # Source-table schema, not the result-set schema: the structural
            # rubric checks (columns/where/joins/subquery) need the tables the
            # SQL was written against.
            table_schema = _get_table_schema_prompt(state.get("tables", []) or [])
            evaluator = SimpleSQLEvaluator(llm)
            result = evaluator.evaluate(question, sql_query, schema_info, data_sample, table_schema=table_schema)
        except Exception as e:  # never block the answer on evaluator failure
            log(f"Confidence evaluation failed: {str(e)}")
            return {}
        log(f"SQL confidence={result.score:.2f} reasons={result.reasons}")
        return {"sql_confidence": result.score, "confidence_reasons": list(result.reasons)}

    def _capture_golden_sql(state: SQLGraphState) -> None:
        """Dual-write an approved SQL: S3 vector store + durable YAML (mandatory)."""
        try:
            cfg = config.get()
        except ValueError:
            return
        if not bool(getattr(cfg, "enable_golden_sql", False)):
            return
        question = state.get("rewrite_question", "")
        sql_query = state.get("sql", "").strip()
        tables = [d["table"] for d in state.get("tables", []) if isinstance(d, dict) and d.get("table")]
        if not question or not sql_query:
            return
        # 1) runtime vector store (S3) — under the store's own lock.
        try:
            store = get_learned_sql_store()
            if store is not None:
                store.add_golden_sql(question, sql_query, tables)
        except Exception as e:
            log(f"Golden SQL vector write failed: {str(e)}")
        # 2) durable YAML append (de-dup, not overwrite) — both writes are mandatory.
        try:
            cfg.catalog_store.append_sql_example(question, sql_query, tables, source="golden")
        except Exception as e:
            log(f"Golden SQL durable write failed: {str(e)}")

    def confidence_gate_node(state: SQLGraphState) -> dict:
        """Interrupt for human review when confidence is below threshold.

        Reuses the ask_human interrupt channel (buttons approve/reject/edit).
        Returns the human decision (and edited SQL on 'edit').
        """
        try:
            cfg = config.get()
            enabled = bool(getattr(cfg, "enable_confidence_gate", False))
            threshold = float(getattr(cfg, "sql_confidence_threshold", 0.7))
        except ValueError:
            enabled, threshold = False, 0.7
        score = state.get("sql_confidence", 1.0)
        if not enabled or score is None or score >= threshold:
            _capture_golden_sql(state)
            _maybe_capture_pattern(state)
            return {"human_sql_decision": "approve"}
        reasons = state.get("confidence_reasons", [])
        feedback = interrupt(
            {
                "text": f"Low-confidence SQL ({score:.2f}). Reasons: {'; '.join(reasons) or 'n/a'}. Approve?",
                "buttons": ["approve", "reject", "edit"],
                "sql": state.get("sql", ""),
            }
        )
        decision = feedback if isinstance(feedback, str) else (feedback or {}).get("decision", "approve")
        if decision == "edit":
            edited = feedback.get("sql") if isinstance(feedback, dict) else None
            if edited:
                return {"human_sql_decision": "edit", "sql": edited}
            # 'edit' without a replacement SQL would re-execute the same SQL,
            # score low again and interrupt forever; degrade to reject so the
            # graph regenerates instead of looping.
            log("Confidence gate: 'edit' without SQL payload, degrading to reject")
            decision = "reject"
        if decision == "approve":
            _capture_golden_sql(state)
            # Human approval overrides the score threshold for pattern capture.
            _maybe_capture_pattern(state, human_approved=True)
        return {"human_sql_decision": decision}

    return (
        generate_sql_node,
        execute_sql_node,
        regenerate_sql_node,
        generate_visualization_node,
        score_sql_node,
        confidence_gate_node,
    )


def should_execute_sql(state: SQLGraphState) -> str:
    """Conditional edge function to determine if SQL should be executed.

    Args:
        state (SQLGraphState): Current state

    Returns:
        str: Next node name - "execute_sql" if SQL is generated, "end" if done
    """
    sql = state.get("sql", "").strip()
    if not sql or sql.lower() == "null":
        return "end"
    else:
        return "execute_sql"
