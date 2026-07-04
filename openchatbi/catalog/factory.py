import logging
import os
from typing import Any

from openchatbi.catalog.catalog_loader import load_catalog_from_data_warehouse
from openchatbi.catalog.catalog_store import CatalogStore
from openchatbi.catalog.store.database import DatabaseCatalogStore
from openchatbi.catalog.store.file_system import FileSystemCatalogStore

logger = logging.getLogger(__name__)


# Factory function for creating CatalogStore instances
def create_catalog_store(
    store_type: str, auto_load: bool = True, data_warehouse_config: dict[str, Any] | None = None, **kwargs
) -> CatalogStore:
    """
    Create a CatalogStore instance

    Args:
        store_type (str): Storage type, supports 'file_system' and 'database'
        auto_load (bool): Whether to autoload from database if catalog data doesn't exist
        data_warehouse_config (dict): Data warehouse configuration dictionary. This is
            used only to build the data warehouse execution engine, and is never reused
            as the catalog persistence database connection.
        **kwargs: Other parameters.
            For 'file_system': data_path.
            For 'database': connection_string, engine, auto_create_schema, echo.

    Returns:
        CatalogStore: CatalogStore instance

    Raises:
        ValueError: If the storage type is not supported
    """
    if store_type == "file_system":
        data_path = kwargs.get("data_path", "data")
        # convert relative path to absolute path
        if not data_path.startswith("/"):
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), data_path)
        catalog_store: CatalogStore = FileSystemCatalogStore(data_path, data_warehouse_config or {})

        # Check if autoload is enabled and if catalog files are missing
        if auto_load:
            _auto_load_catalog_if_needed(catalog_store)

        return catalog_store
    elif store_type == "database":
        catalog_store = DatabaseCatalogStore(
            connection_string=kwargs.get("connection_string"),
            engine=kwargs.get("engine"),
            data_warehouse_config=data_warehouse_config or {},
            auto_create_schema=kwargs.get("auto_create_schema", True),
            echo=kwargs.get("echo", False),
        )

        # When the catalog DB is empty, optionally load the catalog from the data warehouse.
        if auto_load:
            _auto_load_catalog_if_needed(catalog_store)

        return catalog_store
    else:
        raise ValueError(f"Unsupported storage type: {store_type}")


def _auto_load_catalog_if_needed(catalog_store: CatalogStore) -> None:
    """
    Autoload catalog from data warehouse if catalog files are missing or empty

    Args:
        catalog_store (CatalogStore): The catalog store instance
    """

    # Check if catalog store has existing data using the store's own check_exists method
    if not catalog_store.check_exists():
        logger.info("Catalog files missing or empty, attempting to load from data warehouse...")

        try:
            # Get data warehouse config from loaded configuration
            data_warehouse_config = catalog_store.get_data_warehouse_config()
            if not data_warehouse_config:
                logger.warning("No data warehouse configuration found, skipping autoload")
                return

            warehouse_uri = data_warehouse_config.get("uri")
            if not warehouse_uri:
                logger.warning("No data warehouse URI found in configuration, skipping autoload")
                return

            # load catalog from data warehouse
            success = load_catalog_from_data_warehouse(catalog_store)

            if success:
                logger.info("Successfully loaded catalog from data warehouse")
            else:
                logger.error("Failed to load catalog from data warehouse")
                raise Exception("Failed to load catalog from data warehouse")

        except Exception as e:
            logger.warning(f"Autoload from data warehouse failed: {e}")
            raise Exception("Failed to load catalog from data warehouse") from e
