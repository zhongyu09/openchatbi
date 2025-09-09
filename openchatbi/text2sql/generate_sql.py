import datetime
from collections.abc import Callable
from typing import Any

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
from openchatbi.utils import get_text_from_content, log

COLUMN_PROMPT_TEMPLATE = """### Columns
Column(Name, Type, Display Name, Description):
[
{}
]
"""


def create_sql_nodes(llm: BaseChatModel, catalog: CatalogStore, dialect: str) -> tuple[Callable, Callable, Callable]:
    """Creates the three SQL processing nodes for LangGraph.

    Args:
        llm (BaseChatModel): The language model to use for SQL generation.
        catalog (CatalogStore): The catalog store containing schema information.
        dialect (str): The SQL dialect to use (e.g., 'presto', 'mysql').

    Returns:
        tuple: Three node functions (generate_sql_node, execute_sql_node, regenerate_sql_node)
    """

    def _get_column_prompt(column: dict[str, Any]) -> str:
        alias_prompt = f"alias({column['alias']})" if "alias" in column and column["alias"] else ""
        return (
            f"""    Column("{column['column_name']}", {column['type']}, {column['display_name']},"""
            f""" "{alias_prompt}{column['description']}"),"""
        )

    def _get_table_schema_prompt(tables: list[dict[str, Any]]) -> str:
        """Generates a prompt string for table schemas, including table description,
        columns, derived metrics and rules when writting SQL

        Args:
            tables (List[Dict[str, Any]]): List of tables with their columns.

        Returns:
            str: Formatted table schema prompt string.
        """
        schema_prompt = []
        for table_dict in tables:
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

    def _get_relevant_sql_examples_prompt(question, tables: list[str]) -> str:
        """Retrieves relevant SQL examples based on the question and selected tables.

        Args:
            question (str): The natural language question.
            tables (List[str]): List of selected table names.

        Returns:
            str: Formatted string of relevant SQL examples.
        """
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

    def _execute_sql(sql: str):
        """Executes the generated SQL query and returns the result.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            str: The result of the SQL query in CSV format.
        """
        with catalog.get_sql_engine().connect() as connection:
            result = connection.execute(text(sql))

            # Fetch all rows from the result
            rows = result.fetchall()

            # Get column names
            columns = result.keys()

            # Format as CSV
            # Create CSV header
            csv_lines = [",".join(str(col) for col in columns)]

            # Add data rows
            for row in rows:
                csv_lines.append(",".join(str(value) if value is not None else "" for value in row))

            connection.commit()
            return "\n".join(csv_lines)

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
        tables = state["tables"]
        system_prompt = (
            get_text2sql_dialect_prompt_template(dialect)
            .replace("[table_schema]", _get_table_schema_prompt(tables))
            .replace("[examples]", _get_relevant_sql_examples_prompt(question, tables))
            .replace("[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format))
        )

        user_prompt = f"""Generate a SQL query for the question: {question}"""
        messages = [SystemMessage(system_prompt)] + list(state["messages"]) + [HumanMessage(user_prompt)]

        response = llm.invoke(messages)
        response_content = get_text_from_content(response.content)
        sql_query = response_content.replace("```sql", "").replace("```", "").strip()

        if not sql_query:
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
            execute_result = _execute_sql(sql_query)
            result = f"```sql\n{sql_query}\n```\nSQL Result:\n```csv\n{execute_result}\n```"
            return {"sql_execution_result": SQL_SUCCESS, "data": execute_result, "messages": [AIMessage(result)]}
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

    return generate_sql_node, execute_sql_node, regenerate_sql_node


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
