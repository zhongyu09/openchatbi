"""Tools for searching knowledge bases and schema information."""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.catalog.schema_retrival import col_dict, column_tables_mapping, get_relevant_columns
from openchatbi.text2sql.data import get_learned_sql_store
from openchatbi.utils import log


class SearchKnowledgeInput(BaseModel):
    """Input schema for knowledge search tool."""

    reasoning: str = Field(description="Reason for using this search tool")
    query_list: list[str] = Field(
        min_length=1, max_length=5, description="Query terms to search (max 5, avoid duplicates)"
    )
    knowledge_bases: list[Literal["columns", "business", "sql_examples"]] = Field(
        min_length=1,
        max_length=3,
        description="""Knowledge bases to search, options are:
            - `"columns"`: Field-level knowledge such as descriptions, aliases, categories, dimensions and metrics.
            - `"business"`: The business knowledge.
            - `"sql_examples"`: Validated example Question->SQL pairs (golden / learned).""",
    )


@tool("search_knowledge", args_schema=SearchKnowledgeInput, return_direct=False, infer_schema=True)
def search_knowledge(
    reasoning: str, query_list: list[str], knowledge_bases: list[Literal["columns", "business", "sql_examples"]]
) -> dict[str, object]:
    """Search relevant knowledge from knowledge bases.
    Returns:
        Dict[str, object]: Search results for each knowledge base.
    """
    log(f"Search knowledge, query_list={query_list}, knowledge_bases={knowledge_bases}, reasoning={reasoning}")
    final_results: dict[str, object] = {"warnings": []}
    warnings: list[str] = []
    if "columns" in knowledge_bases:
        columns = _search_column_knowledge_from_catalog(query_list)
        final_results["columns"] = columns
        if not columns:
            warnings.append("No relevant field-level knowledge found in columns.")
    if "business" in knowledge_bases:
        business_result, warning = _search_business_knowledge(query_list)
        final_results["business"] = business_result
        if warning:
            warnings.append(warning)
    if "sql_examples" in knowledge_bases:
        sql_examples_result, warning = _search_sql_examples(query_list)
        final_results["sql_examples"] = sql_examples_result
        if warning:
            warnings.append(warning)
    final_results["warnings"] = warnings
    return final_results


class SearchSchemaInput(BaseModel):
    """Input schema for structured schema discovery."""

    reasoning: str = Field(description="Reason for searching schema")
    query_list: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Business terms, entities, metrics, and fields to discover schema for",
    )
    dimensions: list[str] = Field(default_factory=list, description="Known dimension terms to prioritize")
    metrics: list[str] = Field(default_factory=list, description="Known metric terms to prioritize")
    max_tables: int = Field(default=5, ge=1, description="Maximum number of candidate tables to return")
    include_columns: bool = Field(default=True, description="Include detailed matched column metadata")


@tool("search_schema", args_schema=SearchSchemaInput, return_direct=False, infer_schema=True)
def search_schema(
    reasoning: str,
    query_list: list[str],
    dimensions: list[str] | None = None,
    metrics: list[str] | None = None,
    max_tables: int = 5,
    include_columns: bool = True,
) -> dict[str, object]:
    """Discover candidate tables and fields for a data question.

    This tool is for structured schema discovery. Use search_knowledge for
    field/business knowledge and show_schema when the table names are already known.
    """
    log(
        "Search schema, "
        f"query_list={query_list}, dimensions={dimensions}, metrics={metrics}, max_tables={max_tables}, "
        f"reasoning={reasoning}"
    )
    return _find_relevant_schema(
        query_list=query_list,
        dimensions=dimensions or [],
        metrics=metrics or [],
        max_tables=max_tables,
        include_columns=include_columns,
    )


class ShowSchemaInput(BaseModel):
    """Input schema for show schema tool."""

    reasoning: str = Field(description="Reason for showing schema")
    tables: list[str] = Field(min_length=1, max_length=5, description="Full table names to show (max 5)")


@tool("show_schema", args_schema=ShowSchemaInput, return_direct=False, infer_schema=True)
def show_schema(reasoning: str, tables: list[str]) -> dict[str, object]:
    """Show table schemas including description, columns, and derived metrics.
    Returns:
        dict[str, object]: Schema information and missing table names.
    """
    log(f"Show schema, tables={tables}, reasoning={reasoning}")
    result = _list_table_from_catalog(tables)
    return result


def _search_column_knowledge_from_catalog(query_list: list[str]) -> list[dict[str, object]]:
    """Search field-level knowledge from catalog based on query terms."""
    relevant_column_set = set()
    for keywords in query_list:
        relevant_columns = get_relevant_columns(keywords.split(" "), keywords.split(" "), keywords.split(" "))
        relevant_column_set.update(relevant_columns)
    return [
        _column_public_info(col_dict[column_name])
        for column_name in sorted(relevant_column_set)
        if column_name in col_dict
    ]


def _find_relevant_schema(
    query_list: list[str],
    dimensions: list[str] | None = None,
    metrics: list[str] | None = None,
    max_tables: int = 5,
    include_columns: bool = True,
) -> dict[str, object]:
    """Find candidate tables and schema details for a query."""
    dimensions = dimensions or []
    metrics = metrics or []
    catalog_store = config.get().catalog_store
    warnings: list[str] = []

    relevant_columns = get_relevant_columns(query_list, dimensions, metrics)
    if not relevant_columns:
        warnings.append("No relevant columns matched the schema query terms.")
    table_to_matches: dict[str, list[str]] = {}
    for column_name in relevant_columns:
        for table_name in column_tables_mapping.get(column_name, []):
            table_to_matches.setdefault(table_name, []).append(column_name)
    if relevant_columns and not table_to_matches:
        warnings.append("Relevant columns were found but no related tables were mapped.")

    candidates = []
    for table_name, matched_columns in table_to_matches.items():
        table_info = catalog_store.get_table_information(table_name)
        if not table_info:
            continue
        all_columns = catalog_store.get_column_list(table_name)
        matched_set = set(matched_columns)
        table_columns = {column["column_name"]: column for column in all_columns}
        matched_details = [
            _column_public_info(table_columns[column_name])
            for column_name in matched_columns
            if column_name in table_columns
        ]
        date_columns = [_column_name(column) for column in all_columns if _is_date_time_column(column)]
        metric_columns = [_column_name(column) for column in all_columns if column.get("category") == "metric"]
        dimension_columns = [_column_name(column) for column in all_columns if column.get("category") == "dimension"]
        score = len(matched_set)
        candidates.append(
            {
                "table": table_name,
                "description": table_info.get("description", ""),
                "selection_rule": table_info.get("selection_rule", ""),
                "matched_columns": sorted(matched_set),
                "date_columns": date_columns,
                "metric_columns": metric_columns,
                "dimension_columns": dimension_columns,
                "columns": matched_details if include_columns else [],
                "match_reason": _build_match_reason(table_name, matched_set, date_columns, metric_columns),
                "_score": score,
            }
        )

    candidates.sort(key=lambda item: (-item["_score"], item["table"]))
    for candidate in candidates:
        candidate.pop("_score", None)
    if len(candidates) > max_tables:
        warnings.append(f"Returned top {max_tables} tables out of {len(candidates)} candidates.")
    if not candidates:
        warnings.append("No schema candidates found for the provided query terms.")

    unmatched_terms = query_list if not candidates else []
    return {
        "candidates": candidates[:max_tables],
        "matched_column_count": len(relevant_columns),
        "unmatched_terms": unmatched_terms,
        "warnings": warnings,
    }


def _list_table_from_catalog(tables: list[str]) -> dict[str, object]:
    """Get table information from catalog."""
    schemas = []
    missing_tables = []
    warnings = []
    catalog_store = config.get().catalog_store

    for table_name in tables:
        table_info = catalog_store.get_table_information(table_name)
        if not table_info:
            missing_tables.append(table_name)
            continue
        columns = catalog_store.get_column_list(table_name)
        schemas.append(
            {
                "table": table_name,
                "description": table_info.get("description", ""),
                "selection_rule": table_info.get("selection_rule", ""),
                "sql_rule": table_info.get("sql_rule", ""),
                "derived_metric": table_info.get("derived_metric", ""),
                "columns": [_column_public_info(column) for column in columns],
                "date_columns": [_column_name(column) for column in columns if _is_date_time_column(column)],
                "metric_columns": [_column_name(column) for column in columns if column.get("category") == "metric"],
                "dimension_columns": [
                    _column_name(column) for column in columns if column.get("category") == "dimension"
                ],
            }
        )
    if missing_tables:
        warnings.append(f"Some tables were not found in catalog: {missing_tables}")
    if not schemas:
        warnings.append("No schemas were returned for the provided table list.")
    return {"schemas": schemas, "missing_tables": missing_tables, "warnings": warnings}


def _column_name(column: dict[str, object]) -> str:
    return str(column.get("column_name", ""))


def _column_public_info(column: dict[str, object]) -> dict[str, object]:
    """Return stable field-level metadata for tools."""
    return {
        "column_name": column.get("column_name", ""),
        "display_name": column.get("display_name", ""),
        "category": column.get("category", ""),
        "type": column.get("type", ""),
        "description": column.get("description", ""),
        "alias": column.get("alias", ""),
    }


def _is_date_time_column(column: dict[str, object]) -> bool:
    joined = " ".join(
        str(column.get(key, "")).lower() for key in ("column_name", "display_name", "description", "type")
    )
    return any(marker in joined for marker in ("date", "time", "timestamp", "datetime"))


def _build_match_reason(
    table_name: str, matched_columns: set[str], date_columns: list[str], metric_columns: list[str]
) -> str:
    pieces = [f"{table_name} matches {len(matched_columns)} relevant column(s)"]
    if date_columns:
        pieces.append("has date/time columns for time filtering or grouping")
    if metric_columns:
        pieces.append("has metric columns for aggregation")
    return "; ".join(pieces)


def _search_business_knowledge(query_list: list[str]) -> tuple[str, str | None]:
    """Return configured business knowledge."""
    # not implement yet
    warning = "knowledge base is empty!"
    return "", warning


def _search_sql_examples(query_list: list[str]) -> tuple[str, str | None]:
    """Retrieve top-k validated Question->SQL examples from the learned SQL store (S3)."""
    store = get_learned_sql_store()
    if store is None:
        return (
            "# Relevant SQL Examples:\n(no learned SQL store available)",
            "No learned SQL store available.",
        )
    seen: set[str] = set()
    blocks: list[str] = []
    for query in query_list:
        for question, sql, _tables in store.retrieve(query, k=5):
            if question in seen:
                continue
            seen.add(question)
            blocks.append(f"<example>\nQ: {question}\nA: {sql}\n</example>")
    warning = None if blocks else "No SQL examples matched query terms."
    body = "\n".join(blocks) if blocks else "(no matching examples)"
    return f"# Relevant SQL Examples:\n{body}", warning
