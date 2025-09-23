"""Helper functions for building column retrieval systems."""

import jieba
from rank_bm25 import BM25Okapi

from openchatbi.llm.llm import embedding_model
from openchatbi.utils import log, create_vector_db


def get_columns_metadata(catalog):
    """Extract column metadata for indexing.

    Args:
        catalog: Catalog store instance.

    Returns:
        tuple: (columns, col_dict, column_tokens, embedding_keys)
    """
    columns = catalog.get_column_list()
    col_dict = {}
    column_tokens = []
    embedding_keys = []
    for column in columns:
        col_dict[column["column_name"]] = column
        text_parts = [
            column.get("column_name", ""),
            column.get("display_name", ""),
            column.get("alias", ""),
            column.get("tag", ""),
            column.get("description", ""),
        ]
        text = " ".join(text_parts)
        tokens = [token for token in jieba.cut_for_search(text) if token not in ("_", " ")]
        column_tokens.append(tokens)
        embedding_key = f"{column['column_name']}: {column['display_name']}"
        embedding_keys.append(embedding_key)
    return columns, col_dict, column_tokens, embedding_keys


def build_column_tables_mapping(catalog):
    """Build a mapping of column names to their corresponding table names."""
    column_tables_mapping = {}
    for table_name in catalog.get_table_list():
        for column in catalog.get_column_list(table_name):
            column_name = column["column_name"]
            if column_name not in column_tables_mapping:
                column_tables_mapping[column_name] = []
            column_tables_mapping[column_name].append(table_name)
    return column_tables_mapping


def build_columns_retriever(catalog):
    """Build BM25 and vector retrievers for columns.

    Args:
        catalog: Catalog store instance.

    Returns:
        tuple: (bm25, vector_db, columns, col_dict)
    """
    columns, col_dict, column_tokens, embedding_keys = get_columns_metadata(catalog)

    bm25 = BM25Okapi(column_tokens)

    log("Building vector database for columns...")
    vector_db = create_vector_db(
        embedding_keys,
        embedding_model,
        metadatas=columns,
        collection_name="columns",
        collection_metadata={"hnsw:space": "cosine"},
    )

    return bm25, vector_db, columns, col_dict
