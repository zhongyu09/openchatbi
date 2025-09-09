"""Schema linking module for table and column selection in text2sql."""

from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openchatbi.catalog import CatalogStore
from openchatbi.catalog.schema_retrival import col_dict, column_tables_mapping, get_relevant_columns
from openchatbi.constants import datetime_format
from openchatbi.graph_state import SQLGraphState
from openchatbi.prompts.system_prompt import TABLE_SELECTION_PROMPT_TEMPLATE
from openchatbi.text2sql.data import table_selection_example_dict, table_selection_retriever
from openchatbi.utils import extract_json_from_answer, log


def schema_linking(llm: BaseChatModel, catalog: CatalogStore):
    """Create function for schema linking: select appropriate tables and columns for a question.

    Args:
        llm (BaseChatModel): Language model for table selection.
        catalog (CatalogStore): Catalog store with schema information.

    Returns:
        function: Node function for schema linking based on question.
    """

    def _get_related_tables_and_columns(keywords_list, dimensions, metrics, start_time=None, invalid_table=None):
        """Retrieves tables and columns related to the given keywords, dimensions, and metrics.

        Args:
            keywords_list (list): List of keywords extracted from the question.
            dimensions (list): List of dimensions mentioned in the question.
            metrics (list): List of metrics mentioned in the question.
            start_time (str, optional): Start time for filtering tables.
            invalid_table (list, optional): List of tables to exclude.

        Returns:
            dict: Dictionary mapping table names to their information and related columns.
        """
        # 1. Get the top similar columns
        relevant_columns = get_relevant_columns(keywords_list, dimensions, metrics)

        # 2. Get all the related tables
        candidate_tables = set()
        for column in relevant_columns:
            table_list = column_tables_mapping.get(column, [])
            candidate_tables.update(table_list)
        if start_time:
            try:
                start_time = datetime.strptime(start_time, datetime_format)
            except ValueError:
                start_time = None

        # 3. Get all the table's related column
        related_table_column_dict = {}
        for table_name in candidate_tables:
            if table_name in invalid_table:
                continue
            table_info = catalog.get_table_information(table_name)
            if not table_info:
                continue
            if start_time and "start_time" in table_info:
                if datetime.strptime(table_info.get("start_time"), datetime_format) > start_time:
                    continue
            columns = []
            for column_name in relevant_columns:
                column_dict = col_dict[column_name].copy()
                if table_name not in column_tables_mapping.get(column_name, []):
                    continue
                columns.append(column_dict)
            related_table_column_dict[table_name] = (table_info, columns)

        return related_table_column_dict

    def _example_retrieval(query, candidate_tables):
        """Retrieves example questions and their selected tables that match the candidate tables.

        Args:
            query (str): The natural language question.
            candidate_tables (list): List of candidate table names.

        Returns:
            dict: Dictionary mapping example questions to their selected tables.
        """
        similar_questions = table_selection_retriever.invoke(query)
        valid_examples = {}
        for question_doc in similar_questions:
            question = question_doc.page_content
            if not question:
                continue
            expected_tables = table_selection_example_dict[question]
            expected_tables = [table for table in expected_tables if table in candidate_tables]
            if expected_tables:
                valid_examples[question] = expected_tables
        return valid_examples

    def _build_table_selection_prompt(related_table_column_dict, similar_examples):
        """Builds a prompt for table selection based on related tables and examples.

        Args:
            related_table_column_dict (dict): Dictionary of tables with their information and columns.
            similar_examples (dict): Dictionary of example questions and their selected tables.

        Returns:
            str: Formatted prompt for table selection.
        """
        similar_examples = [
            f"- Question: {example}   Selected Tables: [{','.join(selected_tables)}]"
            for example, selected_tables in similar_examples.items()
        ]

        table_column_descs = []
        for table_name, (table_info, columns) in related_table_column_dict.items():
            columns_desc = "\n".join(
                [
                    f"- {column['category']}({column['column_name']}, {column['display_name']}, \"{column['description']}\")"
                    for column in columns
                ]
            )
            desc_part = f"\n### Table Description: \n{table_info['description']}"
            rule_part = f"\n### Rule: \n{table_info.get('selection_rule')}" if table_info.get("selection_rule") else ""
            table_desc = (
                f"\n## Table: {table_name} {desc_part} {rule_part}"
                "\n### Columns: \nCategory(Name, Display Name, Description): "
                f"\n{columns_desc}"
                ""
            )
            table_column_descs.append(table_desc)

        # Build the LLM prompt
        prompt = TABLE_SELECTION_PROMPT_TEMPLATE.replace("[tables]", "\n\n".join(table_column_descs)).replace(
            "[examples]", "\n".join(similar_examples)
        )
        return prompt

    def _verify_table(selected_tables, candidate_tables):
        """Verifies that selected tables are valid candidates.

        Args:
            selected_tables (list): List of tables selected by the model.
            candidate_tables (list): List of candidate tables.

        Returns:
            bool: True if all selected tables are valid candidates.
        """
        if not selected_tables:
            return False
        for table in selected_tables:
            if table.get("table") not in candidate_tables:
                return False
        return True

    def _call_llm_select(llm: BaseChatModel, system_prompt, messages, question, candidate_tables):
        """Calls the language model to select appropriate tables for the question.

        Retries up to 3 times if the LLM's answer is invalid.

        Args:
            llm (BaseChatModel): The language model to use.
            system_prompt (str): The system prompt for table selection.
            messages (list): List of previous messages.
            question (str): The natural language question.
            candidate_tables (list): List of candidate tables.

        Returns:
            dict: Dictionary containing selected tables.
        """
        log("Selecting appropriate tables...")
        # print(f"candidate_tables: {candidate_tables}")
        prompt = f"""Please select the appropriate tables for the question: {question}"""
        messages.append(HumanMessage(prompt))
        retry_flag = True
        retry_cnt = 1
        while retry_flag:
            try:
                log("Ask LLM to select the table...")
                # print("_call_llm_select")
                # print(messages)
                response = llm.invoke([SystemMessage(system_prompt)] + messages)
                result = extract_json_from_answer(response.content)
                selected_tables = result.get("tables")
                log(result)
                if _verify_table(selected_tables, candidate_tables):
                    return {"tables": selected_tables}
                else:
                    messages.append(
                        HumanMessage(
                            f'The selected table {",".join([table.get("table") for table in result.get("tables")])} is not valid. '
                            f"Do not select this table, please try again."
                        )
                    )
                retry_cnt += 1
                if retry_cnt > 3:
                    retry_flag = False
                if retry_flag:
                    log(
                        f"The selected table {','.join([table.get('table') for table in result.get('tables')])} is not in the candidate tables."
                    )
                    log("Retry Table Selection...")

            except Exception as e:
                log(str(e))
                retry_cnt += 1
                if retry_cnt > 3:
                    retry_flag = False
        return {}

    def _select(state: SQLGraphState) -> dict:
        if not state.get("rewrite_question"):
            log("Missing rewrite question, skipping schema linking.")
            return {}

        messages = state["messages"]
        question = state["rewrite_question"]
        info_entities = state["info_entities"]
        keywords_list = info_entities.get("keywords", [])
        dimensions = info_entities.get("dimensions", [])
        metrics = info_entities.get("metrics", [])
        start_time = info_entities.get("start_time")

        invalid_table = []
        log("Retrieving related table schema...")
        # 1. Get related tables and columns
        related_table_column_dict = _get_related_tables_and_columns(
            keywords_list, dimensions, metrics, start_time, invalid_table
        )
        candidate_tables = related_table_column_dict.keys()

        # 2. Get the similar examples
        similar_examples = _example_retrieval(" ".join(keywords_list), related_table_column_dict.keys())

        # 3. Build tables prompt
        system_prompt = _build_table_selection_prompt(related_table_column_dict, similar_examples)

        # 4. Call LLM to select the table
        return _call_llm_select(llm, system_prompt, messages, question, candidate_tables)

    return _select
