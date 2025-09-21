import os

from openchatbi.catalog.entry import catalog_store
from openchatbi.text2sql.text2sql_utils import init_sql_example_retriever, init_table_selection_example_dict

# Skip init during documentation build
if not os.environ.get("SPHINX_BUILD"):
    sql_example_retriever, sql_example_dicts = init_sql_example_retriever(catalog_store)
    table_selection_retriever, table_selection_example_dict = init_table_selection_example_dict(catalog_store)
else:
    sql_example_retriever, sql_example_dicts = None, {}
    table_selection_retriever, table_selection_example_dict = None, {}
