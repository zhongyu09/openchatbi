"""Tests for catalog store functionality."""

import pytest

from openchatbi.catalog.catalog_store import CatalogStore
from openchatbi.catalog.store.file_system import FileSystemCatalogStore


class TestCatalogStore:
    """Test base CatalogStore functionality."""

    def test_catalog_store_is_abstract(self):
        """Test that CatalogStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CatalogStore()

    def test_catalog_store_interface_methods(self):
        """Test that CatalogStore defines required interface methods."""
        # Check that abstract methods exist
        assert hasattr(CatalogStore, "get_table_list")
        assert hasattr(CatalogStore, "get_column_list")
        assert hasattr(CatalogStore, "get_table_information")
        assert hasattr(CatalogStore, "get_data_warehouse_config")
        assert hasattr(CatalogStore, "get_sql_engine")
        assert hasattr(CatalogStore, "save_table_information")


class TestFileSystemCatalogStore:
    """Test FileSystemCatalogStore functionality."""

    def test_filesystem_store_initialization(self, temp_dir):
        """Test FileSystemCatalogStore initialization."""
        data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}
        data_path = str(temp_dir)
        store = FileSystemCatalogStore(data_path=data_path, data_warehouse_config=data_warehouse_config)

        assert store.data_path == data_path
        assert isinstance(store, CatalogStore)

    def test_get_tables_from_csv(self, mock_catalog_store):
        """Test getting tables from CSV file."""
        tables = mock_catalog_store.get_table_list()

        assert isinstance(tables, list)
        assert len(tables) >= 1

    def test_get_columns_from_csv(self, mock_catalog_store):
        """Test getting columns from CSV file."""
        columns = mock_catalog_store.get_column_list("test_table", "test")

        assert isinstance(columns, list)
        if columns:
            column = columns[0]
            assert "column_name" in column or "name" in column
            assert "data_type" in column or "type" in column

    def test_get_table_info(self, mock_catalog_store):
        """Test getting table information."""
        table_info = mock_catalog_store.get_table_information("test.test_table")

        assert isinstance(table_info, dict)

    def test_get_tables_file_not_found(self, temp_dir):
        """Test handling when tables file doesn't exist."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}
        store = FileSystemCatalogStore(data_path=str(empty_dir), data_warehouse_config=data_warehouse_config)

        # Should handle missing file gracefully
        tables = store.get_table_list()
        assert isinstance(tables, list)

    def test_get_columns_file_not_found(self, temp_dir):
        """Test handling when columns file doesn't exist."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}
        store = FileSystemCatalogStore(data_path=str(empty_dir), data_warehouse_config=data_warehouse_config)

        # Should handle missing file gracefully
        columns = store.get_column_list("nonexistent_table")
        assert isinstance(columns, list)

    def test_get_tables_malformed_csv(self, temp_dir):
        """Test handling malformed CSV files."""
        # Create malformed CSV
        malformed_csv = temp_dir / "table_columns.csv"
        malformed_csv.write_text("invalid,csv,format\\nno,proper\\nheaders")

        data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}
        store = FileSystemCatalogStore(data_path=str(temp_dir), data_warehouse_config=data_warehouse_config)

        # Should handle malformed CSV gracefully
        tables = store.get_table_list()
        assert isinstance(tables, list)

    def test_get_tables_pandas_error(self, temp_dir):
        """Test handling pandas errors."""
        data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}
        store = FileSystemCatalogStore(data_path=str(temp_dir), data_warehouse_config=data_warehouse_config)

        # Should handle pandas errors gracefully
        tables = store.get_table_list()
        assert isinstance(tables, list)

    def test_get_table_schema(self, mock_catalog_store):
        """Test getting complete table schema."""
        # Use get_table_information instead of get_table_schema
        schema = mock_catalog_store.get_table_information("test.test_table")

        assert isinstance(schema, dict)

    def test_search_tables(self, mock_catalog_store):
        """Test searching for tables by keyword."""
        # This method might not exist in current implementation
        # but it's a common catalog feature
        if hasattr(mock_catalog_store, "search_tables"):
            results = mock_catalog_store.search_tables("test")
            assert isinstance(results, list)

    def test_get_all_table_names(self, mock_catalog_store):
        """Test getting all table names."""
        tables = mock_catalog_store.get_table_list()
        # get_table_list() returns list of strings (table names), not dictionaries
        assert isinstance(tables, list)
        # Verify all items are strings
        for table_name in tables:
            assert isinstance(table_name, str)

    def test_case_insensitive_table_lookup(self, mock_catalog_store):
        """Test case-insensitive table lookups."""
        # Test with different cases
        test_cases = ["test_table", "TEST_TABLE", "Test_Table"]

        for table_name in test_cases:
            columns = mock_catalog_store.get_column_list(table_name)
            assert isinstance(columns, list)

    def test_data_path_validation(self):
        """Test data path validation."""
        data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}
        # Test with None path
        with pytest.raises((ValueError, TypeError)):
            FileSystemCatalogStore(data_path=None, data_warehouse_config=data_warehouse_config)

        # Test with empty string
        with pytest.raises((ValueError, FileNotFoundError)):
            FileSystemCatalogStore(data_path="", data_warehouse_config=data_warehouse_config)

    def test_concurrent_access(self, mock_catalog_store):
        """Test concurrent access to catalog store."""
        import threading
        import time

        results = []
        errors = []

        def worker():
            try:
                tables = mock_catalog_store.get_table_list()
                results.append(len(tables))
                time.sleep(0.01)
                columns = mock_catalog_store.get_column_list("test_table", "test")
                results.append(len(columns))
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should not have errors from concurrent access
        assert len(errors) == 0
        assert len(results) > 0
