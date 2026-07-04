"""Tests for catalog migration and the file-system selection-example round-trip fix."""

import pytest
from sqlalchemy import create_engine

from openchatbi.catalog.migrate import migrate_catalog
from openchatbi.catalog.store.database import DatabaseCatalogStore
from openchatbi.catalog.store.file_system import FileSystemCatalogStore


@pytest.fixture
def fs_store(temp_dir):
    data_dir = temp_dir / "fs_catalog"
    data_dir.mkdir()
    return FileSystemCatalogStore(
        data_path=str(data_dir),
        data_warehouse_config={"uri": "sqlite:///:memory:", "database_name": "sales"},
    )


@pytest.fixture
def db_store():
    engine = create_engine("sqlite:///:memory:")
    return DatabaseCatalogStore(engine=engine, auto_create_schema=True)


class TestSelectionExampleRoundTrip:
    """Verify the FileSystemCatalogStore JSON round-trip fix."""

    def test_single_table_round_trip(self, fs_store):
        fs_store.save_table_selection_examples([("Show customers", ["Customers"])])
        assert fs_store.get_table_selection_examples() == [("Show customers", ["Customers"])]

    def test_multi_table_round_trip(self, fs_store):
        examples = [
            ("Show customers", ["Customers"]),
            ("Customer orders", ["Customers", "Orders"]),
        ]
        fs_store.save_table_selection_examples(examples)
        assert fs_store.get_table_selection_examples() == examples

    def test_parse_backward_compat_python_repr(self):
        # Legacy lossy format written as a Python list literal.
        assert FileSystemCatalogStore._parse_selected_tables("['Customers', 'Orders']") == [
            "Customers",
            "Orders",
        ]

    def test_parse_backward_compat_plain_separated(self):
        assert FileSystemCatalogStore._parse_selected_tables("Customers, Orders") == ["Customers", "Orders"]

    def test_parse_json(self):
        assert FileSystemCatalogStore._parse_selected_tables('["Customers", "Orders"]') == [
            "Customers",
            "Orders",
        ]


class TestMigrateCatalog:
    def _populate_source(self, store):
        store.save_table_information(
            "Customers",
            {
                "description": "Customer information table",
                "selection_rule": "Select for customer queries",
                "sql_rule": "Use customer_id as primary key",
                "derived_metric": "count(distinct customer_id)",
            },
            [
                {"column_name": "customer_id", "type": "INTEGER", "description": "PK", "is_common": True},
                {"column_name": "note", "type": "VARCHAR(255)", "description": "note", "is_common": False},
            ],
            database="sales",
        )
        store.save_table_information(
            "Orders",
            {"description": "Orders table"},
            [{"column_name": "order_id", "type": "INTEGER", "is_common": False}],
            database="sales",
        )
        store.save_table_sql_examples(
            "Customers",
            [{"question": "How many customers?", "answer": "SELECT COUNT(*) FROM Customers"}],
            database="sales",
        )
        store.save_table_selection_examples(
            [("Show customers", ["Customers"]), ("Customer orders", ["Customers", "Orders"])]
        )

    def test_migrate_copies_all_data(self, fs_store, db_store):
        self._populate_source(fs_store)

        stats = migrate_catalog(fs_store, db_store)

        assert stats["tables"] == 2
        assert stats["failed_tables"] == 0
        assert stats["sql_example_tables"] == 1
        assert stats["selection_examples"] == 2

        # Tables & columns
        assert sorted(db_store.get_table_list()) == ["sales.Customers", "sales.Orders"]
        cust_cols = [c["column_name"] for c in db_store.get_column_list("Customers", "sales")]
        assert cust_cols == ["customer_id", "note"]

        # Table information
        info = db_store.get_table_information("Customers", "sales")
        assert info["description"] == "Customer information table"
        assert info["derived_metric"] == "count(distinct customer_id)"

        # SQL examples
        examples = db_store.get_sql_examples("Customers", "sales")
        assert examples == [("How many customers?", "SELECT COUNT(*) FROM Customers", ["sales.Customers"])]

        # Selection examples
        assert db_store.get_table_selection_examples() == [
            ("Show customers", ["Customers"]),
            ("Customer orders", ["Customers", "Orders"]),
        ]

    def test_dry_run_writes_nothing(self, fs_store, db_store):
        self._populate_source(fs_store)
        stats = migrate_catalog(fs_store, db_store, dry_run=True)
        assert stats["tables"] == 2
        assert db_store.get_table_list() == []
        assert db_store.check_exists() is False

    def test_migrate_preserves_common_column_reuse(self, fs_store, db_store):
        common = {"column_name": "created_at", "type": "timestamp", "is_common": True}
        fs_store.save_table_information("A", {"description": "a"}, [common], database="sales")
        fs_store.save_table_information("B", {"description": "b"}, [common], database="sales")

        migrate_catalog(fs_store, db_store)

        assert [c["column_name"] for c in db_store.get_column_list("A", "sales")] == ["created_at"]
        assert [c["column_name"] for c in db_store.get_column_list("B", "sales")] == ["created_at"]
        # Only one reusable common column definition.
        assert len(db_store.get_column_list()) == 1
