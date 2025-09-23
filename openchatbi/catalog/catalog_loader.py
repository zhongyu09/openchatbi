import logging
from typing import Any

from sqlalchemy import MetaData, inspect
from sqlalchemy.engine import Engine

from .catalog_store import CatalogStore

logger = logging.getLogger(__name__)


class DataCatalogLoader:
    """
    The loader to load data catalog from data warehouse metadata and save to catalog store.
    """

    def __init__(self, engine: Engine, include_tables: list[str] | None = None):
        """
        Initialize catalog loader.

        Args:
            engine (Engine): SQLAlchemy engine instance
            include_tables (Optional[List[str]]): List of table names to include, None for all
        """
        self.engine = engine
        self.include_tables = include_tables
        self.metadata = MetaData()
        self.inspector = inspect(engine)

    def get_tables_and_columns(self) -> dict[str, list[dict[str, Any]]]:
        """
        Extract table and column metadata including comments using SQLAlchemy inspector.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping table names to list of column information
        """
        try:
            tables_columns = {}

            # Get all table names
            table_names = self.inspector.get_table_names()

            # Filter to specific tables if configured
            if self.include_tables:
                table_names = [name for name in table_names if name in self.include_tables]

            logger.info(f"Found {len(table_names)} tables to process")

            for table_name in table_names:
                try:
                    # Get column information for the table
                    columns = self.inspector.get_columns(table_name)
                    column_list = []
                    for column in columns:
                        is_common_column = column not in ("id", "name", "type", "status")
                        column_info = {
                            "column_name": column["name"],
                            "display_name": "",
                            "alias": "",
                            "type": str(column["type"]),
                            "category": "",
                            "tag": "",
                            "description": column.get("comment", "") or "",
                            "dimension_table": "",
                            "default": str(column.get("default", "")) if column.get("default") is not None else "",
                            "is_common": is_common_column,
                        }
                        column_list.append(column_info)

                    tables_columns[table_name] = column_list
                    logger.debug(f"Processed table {table_name} with {len(column_list)} columns")

                except Exception as e:
                    logger.error(f"Failed to process table {table_name}: {e}")
                    continue

            logger.info(f"Successfully processed {len(tables_columns)} tables")
            return tables_columns

        except Exception as e:
            logger.error(f"Failed to get tables and columns from data warehouse: {e}")
            return {}

    def get_table_indexes(self, table_name: str) -> list[dict[str, Any]]:
        """
        Get index information for a specific table.

        Args:
            table_name (str): Name of the table

        Returns:
            List[Dict[str, Any]]: List of index information
        """
        try:
            indexes = self.inspector.get_indexes(table_name)
            return indexes
        except Exception as e:
            logger.warning(f"Failed to get indexes for table {table_name}: {e}")
            return []

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        """
        Get foreign key information for a specific table.

        Args:
            table_name (str): Name of the table

        Returns:
            List[Dict[str, Any]]: List of foreign key information
        """
        try:
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            return foreign_keys
        except Exception as e:
            logger.warning(f"Failed to get foreign keys for table {table_name}: {e}")
            return []

    def save_to_catalog_store(
        self, catalog_store: CatalogStore, database_name: str | None = None, update: bool = False
    ) -> bool:
        """
        Extract warehouse metadata and save to catalog store.

        Args:
            catalog_store (CatalogStore): Target catalog store to load data to
            database_name (Optional[str]): Database name in catalog, defaults to 'default'
            update (bool): Update existing catalog store to sync with data warehouse

        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            if database_name is None:
                database_name = "default"

            # Get tables and columns from data warehouse
            tables_columns = self.get_tables_and_columns()

            if not tables_columns:
                logger.warning("No tables found in data warehouse")
                return True

            # Import each table
            success_count = 0
            total_count = len(tables_columns)

            for table_name, columns in tables_columns.items():
                try:
                    # Get table comment if available
                    table_comment = ""
                    try:
                        table_info = self.inspector.get_table_comment(table_name)
                        table_comment = table_info.get("text", "") if table_info else ""
                    except Exception:
                        # Some databases don't support table comments
                        pass

                    table_info = {"description": table_comment, "selection_rule": "", "sql_rule": ""}
                    if catalog_store.save_table_information(table_name, table_info, columns, database_name):
                        success_count += 1
                        logger.info(f"Successfully loaded table: {database_name}.{table_name}")
                    else:
                        logger.error(f"Failed to load table: {database_name}.{table_name}")

                    # init null SQL examples
                    catalog_store.save_table_sql_examples(
                        table_name, [{"question": "null", "answer": "null"}], database_name
                    )

                except Exception as e:
                    logger.error(f"Error loading table {table_name}: {e}")

            # init empty table selection examples
            catalog_store.save_table_selection_examples([("", [])])

            logger.info(f"Load completed: {success_count}/{total_count} tables loaded successfully")
            return success_count == total_count

        except Exception as e:
            logger.error(f"Failed to load data warehouse to catalog store: {e}")
            return False


def load_catalog_from_data_warehouse(catalog_store: CatalogStore) -> bool:
    """
    Load catalog data from data warehouse using SQLAlchemy based on data warehouse config (URI)

    Main entry point for catalog loading.

    Args:
        catalog_store (CatalogStore): Target catalog store

    Returns:
        bool: True if load was successful, False otherwise
    """
    try:
        data_warehouse_config = catalog_store.get_data_warehouse_config()
        database_uri = data_warehouse_config.get("uri")
        include_tables = data_warehouse_config.get("include_tables")
        database_name = data_warehouse_config.get("database_name", "default")
        engine = catalog_store.get_sql_engine()

        loader = DataCatalogLoader(engine, include_tables)
        return loader.save_to_catalog_store(catalog_store, database_name)

    except Exception as e:
        logger.error(f"Failed to import catalog from data warehouse URI {database_uri}: {e}")
        return False
