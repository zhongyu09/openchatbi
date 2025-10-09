import os

from openchatbi import config
from openchatbi.text2sql.text2sql_utils import init_sql_example_retriever, init_table_selection_example_dict

# Skip init during documentation build
if not os.environ.get("SPHINX_BUILD"):
    try:
        _catalog_store = config.get().catalog_store
    except ValueError:
        _catalog_store = None
else:
    _catalog_store = None

if _catalog_store:
    sql_example_retriever, sql_example_dicts = init_sql_example_retriever(_catalog_store)
    table_selection_retriever, table_selection_example_dict = init_table_selection_example_dict(_catalog_store)
else:
    sql_example_retriever, sql_example_dicts = None, {}
    table_selection_retriever, table_selection_example_dict = None, {}
