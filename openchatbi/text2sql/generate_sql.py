import datetime
from collections.abc import Callable
from typing import Any, Tuple, Dict

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, OperationalError, ProgrammingError, TimeoutError

from openchatbi.catalog import CatalogStore
from openchatbi.constants import (
    SQL_EXECUTE_TIMEOUT,
    SQL_NA,
    SQL_SUCCESS,
    SQL_SYNTAX_ERROR,
    SQL_UNKNOWN_ERROR,
    datetime_format,
)
from openchatbi.graph_state import SQLGraphState
from openchatbi.prompts.system_prompt import get_text2sql_dialect_prompt_template
from openchatbi.text2sql.data import sql_example_dicts, sql_example_retriever
from openchatbi.text2sql.visualization import VisualizationService
from openchatbi.utils import get_text_from_content, log

COLUMN_PROMPT_TEMPLATE = """### Columns
Column(Name, Type, Display Name, Description):
[
{}
]
"""


def create_sql_nodes(
    llm: BaseChatModel, catalog: CatalogStore, dialect: str, visualization_mode: str | None = "rule"
) -> tuple[Callable, Callable, Callable, Callable]:
    """Creates the four SQL processing nodes for LangGraph.

    Args:
        llm (BaseChatModel): The language model to use for SQL generation.
        catalog (CatalogStore): The catalog store containing schema information.
        dialect (str): The SQL dialect to use (e.g., 'presto', 'mysql').
        visualization_mode (str | None): Visualization analysis mode ("rule", "llm", or None to skip).

    Returns:
        tuple: Four node functions (generate_sql_node, execute_sql_node, regenerate_sql_node, generate_visualization_node)
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
            single_table_schema_prompt += COLUMN_PROMPT_TEMPLATE.format(
                "\n".join([_get_column_prompt(column) for column in columns])
            )
            single_table_schema_prompt += table_info.get("derived_metric", "")
            single_table_schema_prompt += table_info["sql_rule"]
            schema_prompt.append(single_table_schema_prompt)
        return "\n".join(schema_prompt)

    def _get_relevant_sql_examples_prompt(question, tables_columns: list[dict[str, Any]]) -> str:
        """Retrieves relevant SQL examples based on the question and selected tables.

        Args:
            question (str): The natural language question.
            tables_columns (List[str]): List of selected tables with selected columns.

        Returns:
            str: Formatted string of relevant SQL examples.
        """
        tables = [d["table"] for d in tables_columns]
        relevant_questions = sql_example_retriever.get_relevant_documents(question)
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

    def _analyze_dataframe_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame to understand column types and characteristics."""
        try:
            schema_info = {
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
                    except:
                        schema_info["categorical_columns"].append(col)

            # Calculate unique value counts for categorical columns
            schema_info["unique_counts"] = {}
            for col in schema_info["categorical_columns"]:
                schema_info["unique_counts"][col] = df[col].nunique()

            return schema_info
        except Exception as e:
            return {"error": f"Failed to analyze data schema: {str(e)}"}

    def _execute_sql(sql: str) -> Tuple[dict, str]:
        """Executes the generated SQL query and returns the result with schema analysis.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            Tuple[dict, str]: A tuple containing (schema_info, CSV string).
        """
        with catalog.get_sql_engine().connect() as connection:
            result = connection.execute(text(sql))

            # Fetch all rows from the result
            rows = result.fetchall()

            # Get column names
            columns = list(result.keys())

            # Create DataFrame for analysis
            df = pd.DataFrame(rows, columns=columns)

            # Analyze data schema
            schema_info = _analyze_dataframe_schema(df)

            # Format as CSV
            csv_data = df.to_csv(index=False)

            connection.commit()
            return schema_info, csv_data

    def generate_sql_node(state: SQLGraphState) -> dict:
        """First node: Generates initial SQL query based on the state.

        Args:
            state (SQLGraphState): The current SQL graph state containing the question and tables.

        Returns:
            dict: Updated state with generated SQL query.
        """
        if "rewrite_question" not in state:
            log("Missing rewrite question, skipping SQL generation.")
            return {}
        if "tables" not in state or len(state["tables"]) == 0:
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

        try:
            schema_info, csv_result = _execute_sql(sql_query)
            result = f"```sql\n{sql_query}\n```\nSQL Result:\n```csv\n{csv_result}\n```"
            return {
                "sql_execution_result": SQL_SUCCESS,
                "schema_info": schema_info,
                "data": csv_result,
                "messages": [AIMessage(result)],
            }
        except (OperationalError, TimeoutError) as e:
            log(f"Database connection/timeout error: {str(e)}")
            error_result = (
                f"```sql\n{sql_query}\n```\nDatabase Connection Timeout: {str(e)}\nPlease check database connectivity."
            )
            return {"sql_execution_result": SQL_EXECUTE_TIMEOUT, "messages": [AIMessage(error_result)]}
        except Exception as e:
            error_type = "Unexpected error"
            if isinstance(e, ProgrammingError):
                error_type = "SQL syntax error"
            elif isinstance(e, DatabaseError):
                error_type = "Database error"

            log(f"{error_type}: {str(e)}")

            # Add error to previous errors list
            previous_errors = list(state.get("previous_sql_errors", []))
            previous_errors.append({"sql": sql_query, "error": f"{error_type}: {str(e)}", "error_type": error_type})

            return {
                "sql_execution_result": SQL_UNKNOWN_ERROR if error_type == "Unexpected error" else SQL_SYNTAX_ERROR,
                "previous_sql_errors": previous_errors,
            }

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
                user_prompt += f"\n\nAttempt {i}:\nSQL: {error_info['sql']}\nError: {error_info['error']}"
            user_prompt += "\n\nPlease analyze the errors above and generate a corrected SQL query."

        messages = [SystemMessage(system_prompt)] + list(state["messages"]) + [HumanMessage(user_prompt)]

        response = llm.invoke(messages)
        response_content = get_text_from_content(response.content)
        sql_query = response_content.replace("```sql", "").replace("```", "").strip()

        if not sql_query:
            log(f"Generated SQL query is empty. LLM output: {response.content}")
            error_result = f"Failed to regenerate valid SQL after {retry_count} attempts."
            return {
                "messages": [AIMessage(error_result)],
                "sql": "",
                "sql_retry_count": retry_count,
                "sql_execution_result": SQL_NA,
            }

        return {"sql": sql_query, "sql_retry_count": retry_count, "sql_execution_result": ""}

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
                messages[-1] = AIMessage(current_content + viz_info)

            return {"visualization_dsl": viz_dsl.to_dict(), "messages": messages}
        except Exception as e:
            log(f"Visualization generation error: {str(e)}")
            return {"visualization_dsl": {"error": f"Failed to generate visualization: {str(e)}"}}

    return generate_sql_node, execute_sql_node, regenerate_sql_node, generate_visualization_node


def should_retry_sql(state: SQLGraphState) -> str:
    """Conditional edge function to determine if SQL should be retried.

    Args:
        state (SQLGraphState): Current state

    Returns:
        str: Next node name - "regenerate_sql" if retry needed, "end" if done
    """
    execution_result = state.get("sql_execution_result", "")
    retry_count = state.get("sql_retry_count", 0)
    max_retries = 3

    if execution_result in (SQL_SUCCESS, SQL_EXECUTE_TIMEOUT):
        return "end"
    elif retry_count < max_retries:
        return "regenerate_sql"
    else:
        # Max retries reached or other terminal state
        if retry_count >= max_retries:
            previous_errors = state.get("previous_sql_errors", [])
            if previous_errors:
                last_error = previous_errors[-1]
                error_result = f"```sql\n{last_error['sql']}\n```\n{last_error['error']}\nFailed to generate valid SQL after {max_retries} attempts."
            else:
                error_result = f"Failed to generate valid SQL after {max_retries} attempts."

            # Update state with final error message
            state["messages"] = [AIMessage(error_result)]
            state["sql_execution_result"] = SQL_NA
        return "end"


def should_execute_sql(state: SQLGraphState) -> str:
    """Conditional edge function to determine if SQL should be executed.

    Args:
        state (SQLGraphState): Current state

    Returns:
        str: Next node name - "execute_sql" if SQL is generated, "end" if done
    """
    sql = state.get("sql", "")
    if not sql:
        return "end"
    else:
        return "execute_sql"
