"""Tools for searching knowledge bases and schema information."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from openchatbi.catalog.entry import catalog_store
from openchatbi.catalog.schema_retrival import col_dict, column_tables_mapping, get_relevant_columns
from openchatbi.utils import log


class SearchInput(BaseModel):
    """Input schema for knowledge search tool."""

    reasoning: str = Field(description="Reason for using this search tool")
    query_list: list[str] = Field(description="Query terms to search (max 5, avoid duplicates)")
    knowledge_bases: list[str] = Field(
        description="""Knowledge bases to search, options are:
            - `"columns"`: The description, alias of columns, including dimensions and metrics.
            - `"business"`: The business knowledge."""
    )
    with_table_list: bool = Field(
        description="Include table list for columns (only set to True when user asks about table-column relationships)"
    )


@tool("search_knowledge", args_schema=SearchInput, return_direct=False, infer_schema=True)
def search_knowledge(
    reasoning: str, query_list: list[str], knowledge_bases: list[str], with_table_list: bool = False
) -> dict[str, str]:
    """Search relevant knowledge from knowledge bases.
    Returns:
        Dict[str, str]: Search results for each knowledge base.
    """
    log(f"Search knowledge, query_list={query_list}, knowledge_bases={knowledge_bases}, reasoning={reasoning}")
    final_results = {}
    if "columns" in knowledge_bases:
        column_results = search_column_from_catalog(query_list, with_table_list)
        final_results["columns"] = f"# Relevant Columns and Description:\n{column_results}"
    return final_results


class ShowSchemaInput(BaseModel):
    """Input schema for show schema tool."""

    reasoning: str = Field(description="Reason for showing schema")
    tables: list[str] = Field(description="Full table names to show (max 5)")


@tool("show_schema", args_schema=ShowSchemaInput, return_direct=False, infer_schema=True)
def show_schema(reasoning: str, tables: list[str]) -> list[str]:
    """Show table schemas including description, columns, and derived metrics.
    Returns:
        list[str]: Schema information for each table.
    """
    log(f"Show schema, tables={tables}, reasoning={reasoning}")
    result = list_table_from_catalog(tables)
    return result


def search_column_from_catalog(query_list: list[str], with_table_list: bool) -> str:
    """Search columns from catalog based on query list."""
    relevant_column_set = set()
    for keywords in query_list:
        relevant_columns = get_relevant_columns(keywords.split(" "), keywords.split(" "), keywords.split(" "))
        relevant_column_set.update(relevant_columns)
    column_results = render_column_result(relevant_column_set, with_table_list)
    return "\n".join(column_results)


def list_table_from_catalog(tables: list[str]) -> list[str]:
    """Get table information from catalog."""
    result = []
    for table_name in tables:
        table_info = catalog_store.get_table_information(table_name)
        if not table_info:
            continue
        table_desc = f"Table: `{table_name}` \n# Description: {table_info['description']}\n"
        columns = catalog_store.get_column_list(table_name)
        column_names = [info["column_name"] for info in columns]
        column_results = render_column_result(column_names)
        table_desc += "# Columns:\n"
        table_desc += "\n".join(column_results)
        if table_info.get("derived_metric"):
            table_desc += "## Derived metrics:\n"
            table_desc += table_info["derived_metric"]
        result.append(table_desc)
    return result


def render_column_result(column_list: list[str], with_table_list: bool = False) -> list[str]:
    """Render column information as formatted strings."""
    column_results = []
    for column_name in column_list:
        if column_name not in col_dict:
            continue
        table_list = column_tables_mapping.get(column_name, [])
        column = col_dict[column_name]
        column_desc = (
            f"## {column['column_name']}"
            f"\n- Column Category: {column['category']}"
            f"\n- Display Name: {column['display_name']} "
            f"\n- Description \"{column['description']}\""
        )
        if with_table_list:
            column_desc += f"\n- Related Tables: {table_list}"
        column_results.append(column_desc)
    return column_results
