"""Tests for catalog loader functionality."""

from unittest.mock import Mock, patch

import pytest

from openchatbi.catalog.catalog_loader import DataCatalogLoader, load_catalog_from_data_warehouse


class TestDataCatalogLoader:
    """Test DataCatalogLoader functionality."""

    @pytest.fixture
    def mock_engine(self):
        """Mock SQLAlchemy engine."""
        engine = Mock()
        return engine

    def test_catalog_loader_initialization(self, mock_engine):
        """Test DataCatalogLoader initialization."""
        with patch("openchatbi.catalog.catalog_loader.inspect") as mock_inspect:
            mock_inspect.return_value = Mock()

            loader = DataCatalogLoader(engine=mock_engine, include_tables=["table1", "table2"])

            assert loader.engine == mock_engine
            assert loader.include_tables == ["table1", "table2"]

    def test_catalog_loader_without_include_tables(self, mock_engine):
        """Test DataCatalogLoader without include tables."""
        with patch("openchatbi.catalog.catalog_loader.inspect") as mock_inspect:
            mock_inspect.return_value = Mock()

            loader = DataCatalogLoader(engine=mock_engine, include_tables=None)
            assert loader.include_tables is None

    def test_get_tables_and_columns(self, mock_engine):
        """Test getting tables and columns metadata."""
        # Mock inspector
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["table1", "table2"]
        mock_inspector.get_columns.return_value = [
            {"name": "col1", "type": "VARCHAR(50)", "comment": "Test column", "default": None, "primary_key": False}
        ]

        with patch("openchatbi.catalog.catalog_loader.inspect", return_value=mock_inspector):
            loader = DataCatalogLoader(engine=mock_engine, include_tables=["table1"])
            result = loader.get_tables_and_columns()

            assert "table1" in result
            assert len(result["table1"]) == 1
            assert result["table1"][0]["column_name"] == "col1"

    def test_get_table_indexes(self, mock_engine):
        """Test getting table indexes."""
        mock_inspector = Mock()
        mock_inspector.get_indexes.return_value = [{"name": "idx_test", "column_names": ["col1"]}]

        with patch("openchatbi.catalog.catalog_loader.inspect", return_value=mock_inspector):
            loader = DataCatalogLoader(engine=mock_engine)
            result = loader.get_table_indexes("table1")

            assert len(result) == 1
            assert result[0]["name"] == "idx_test"

    def test_get_foreign_keys(self, mock_engine):
        """Test getting foreign keys."""
        mock_inspector = Mock()
        mock_inspector.get_foreign_keys.return_value = [
            {"name": "fk_test", "constrained_columns": ["col1"], "referred_table": "ref_table"}
        ]

        with patch("openchatbi.catalog.catalog_loader.inspect", return_value=mock_inspector):
            loader = DataCatalogLoader(engine=mock_engine)
            result = loader.get_foreign_keys("table1")

            assert len(result) == 1
            assert result[0]["name"] == "fk_test"

    def test_save_to_catalog_store_success(self, mock_engine):
        """Test saving to catalog store successfully."""
        mock_catalog_store = Mock()
        mock_catalog_store.save_table_information.return_value = True
        mock_catalog_store.save_table_sql_examples.return_value = True
        mock_catalog_store.save_table_selection_examples.return_value = True

        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["table1"]
        mock_inspector.get_columns.return_value = [
            {"name": "col1", "type": "VARCHAR(50)", "comment": "Test column", "default": None, "primary_key": False}
        ]
        mock_inspector.get_table_comment.return_value = {"text": "Test table"}

        with patch("openchatbi.catalog.catalog_loader.inspect", return_value=mock_inspector):
            loader = DataCatalogLoader(engine=mock_engine, include_tables=["table1"])
            result = loader.save_to_catalog_store(mock_catalog_store, "test_db")

            assert result == True
            mock_catalog_store.save_table_information.assert_called()
            mock_catalog_store.save_table_sql_examples.assert_called()
            mock_catalog_store.save_table_selection_examples.assert_called()

    def test_save_to_catalog_store_failure(self, mock_engine):
        """Test handling catalog store save failures."""
        mock_catalog_store = Mock()
        mock_catalog_store.save_table_information.return_value = False

        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["table1"]
        mock_inspector.get_columns.return_value = []

        with patch("openchatbi.catalog.catalog_loader.inspect", return_value=mock_inspector):
            loader = DataCatalogLoader(engine=mock_engine)
            result = loader.save_to_catalog_store(mock_catalog_store)

            assert result == False

    def test_load_catalog_from_data_warehouse(self):
        """Test main entry point for catalog loading."""
        mock_catalog_store = Mock()
        mock_catalog_store.get_data_warehouse_config.return_value = {
            "uri": "test://user@host/db",
            "include_tables": ["table1"],
            "database_name": "test_db",
        }
        mock_catalog_store.get_sql_engine.return_value = Mock()

        with patch("openchatbi.catalog.catalog_loader.DataCatalogLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.save_to_catalog_store.return_value = True
            mock_loader_class.return_value = mock_loader

            result = load_catalog_from_data_warehouse(mock_catalog_store)

            assert result == True
            mock_loader.save_to_catalog_store.assert_called_once()

    def test_error_handling_in_get_tables_and_columns(self, mock_engine):
        """Test error handling in get_tables_and_columns method."""
        mock_inspector = Mock()
        mock_inspector.get_table_names.side_effect = Exception("Database error")

        with patch("openchatbi.catalog.catalog_loader.inspect", return_value=mock_inspector):
            loader = DataCatalogLoader(engine=mock_engine)
            result = loader.get_tables_and_columns()

            assert result == {}
