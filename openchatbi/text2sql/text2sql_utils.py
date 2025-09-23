"""Utility functions for text2sql retrieval systems."""

from openchatbi.llm.llm import embedding_model
from openchatbi.utils import create_vector_db


def init_sql_example_retriever(catalog):
    """Initialize SQL example retriever from catalog.

    Args:
        catalog: Catalog store containing SQL examples.

    Returns:
        tuple: (retriever, sql_example_dict)
    """
    sql_examples = catalog.get_sql_examples()
    sql_example_dict = {q: (sql, table) for q, sql, table in sql_examples}

    texts = list(sql_example_dict.keys())
    vector_db = create_vector_db(
        texts,
        embedding_model,
        collection_name="text2sql",
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"distance_metric": "cosine", "fetch_k": 30, "k": 10}
    )
    return retriever, sql_example_dict


def init_table_selection_example_dict(catalog):
    """Initialize table selection example retriever from catalog.

    Args:
        catalog: Catalog store containing table selection examples.

    Returns:
        tuple: (retriever, table_selection_example_dict)
    """
    sql_examples = catalog.get_table_selection_examples()
    table_selection_example_dict = dict((q, tables) for q, tables in sql_examples)

    texts = list(table_selection_example_dict.keys())
    if not texts:
        texts = [""]  # Empty text as fallback

    vector_db = create_vector_db(
        texts,
        embedding_model,
        collection_name="table_selection_example",
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"distance_metric": "cosine", "fetch_k": 30, "k": 10}
    )
    return retriever, table_selection_example_dict
