import os
import threading

from openchatbi import config
from openchatbi.text2sql.text2sql_utils import (
    LearnedSQLStore,
    _init_sql_example_retriever,
    _init_table_selection_example_dict,
)

# Skip init during documentation build
if not os.environ.get("SPHINX_BUILD"):
    try:
        _catalog_store = config.get().catalog_store
    except ValueError:
        _catalog_store = None
else:
    _catalog_store = None

if _catalog_store:
    sql_example_retriever, sql_example_dicts, sql_example_vector_db = _init_sql_example_retriever(
        _catalog_store, config.get().vector_db_path
    )
    learned_sql_store = LearnedSQLStore(sql_example_vector_db, sql_example_dicts, threading.Lock())
    table_selection_retriever, table_selection_example_dict = _init_table_selection_example_dict(
        _catalog_store, config.get().vector_db_path
    )
else:
    sql_example_retriever, sql_example_dicts, sql_example_vector_db = None, {}, None
    learned_sql_store: LearnedSQLStore | None = None
    table_selection_retriever, table_selection_example_dict = None, {}


def get_learned_sql_store():
    """Return the module-level LearnedSQLStore singleton (or None if not initialized)."""
    return learned_sql_store
