"""Tests for the SQLAlchemy-backed DatabaseCatalogStore."""

import pytest
from sqlalchemy import create_engine

from openchatbi.catalog.catalog_store import CatalogStore
from openchatbi.catalog.factory import create_catalog_store
from openchatbi.catalog.store.database import DatabaseCatalogStore
from openchatbi.catalog.store.file_system import FileSystemCatalogStore


@pytest.fixture
def db_store() -> DatabaseCatalogStore:
    """In-memory SQLite DatabaseCatalogStore.

    Uses a shared engine so the in-memory database persists across the sessions
    created by each store method.
    """
    engine = create_engine("sqlite:///:memory:")
    return DatabaseCatalogStore(engine=engine, auto_create_schema=True)


def _sample_columns() -> list[dict]:
    return [
        {
            "column_name": "customer_id",
            "display_name": "Customer ID",
            "alias": "cust_id",
            "type": "INTEGER",
            "category": "identifier",
            "tag": "customer",
            "description": "Unique identifier for customers",
            "is_common": True,
        },
        {
            "column_name": "special_note",
            "display_name": "Special Note",
            "type": "VARCHAR(255)",
            "category": "attribute",
            "description": "Table-specific note",
            "is_common": False,
        },
    ]


def _sample_information() -> dict:
    return {
        "description": "Customer information table",
        "selection_rule": "Select for customer queries",
        "sql_rule": "Use customer_id as primary key",
        "derived_metric": "count(distinct customer_id) as customer_count",
    }


# --------------------------------------------------------------------------
# 5.2 Initialization
# --------------------------------------------------------------------------
class TestInitialization:
    def test_init_with_connection_string(self):
        store = DatabaseCatalogStore(connection_string="sqlite:///:memory:")
        assert isinstance(store, CatalogStore)
        assert store.check_exists() is False

    def test_init_with_engine(self):
        engine = create_engine("sqlite:///:memory:")
        store = DatabaseCatalogStore(engine=engine)
        assert isinstance(store, CatalogStore)

    def test_init_requires_engine_or_connection_string(self):
        with pytest.raises(ValueError):
            DatabaseCatalogStore()

    def test_auto_create_schema_false_skips_table_creation(self):
        engine = create_engine("sqlite:///:memory:")
        store = DatabaseCatalogStore(engine=engine, auto_create_schema=False)
        # Without schema, querying raises an operational error surfaced as False.
        assert store.check_exists() is False

    def test_data_warehouse_config_distinct_from_catalog_db(self):
        engine = create_engine("sqlite:///:memory:")
        dw_config = {"uri": "sqlite:///:memory:", "database_name": "dw"}
        store = DatabaseCatalogStore(engine=engine, data_warehouse_config=dw_config)
        assert store.get_data_warehouse_config() == dw_config
        # The data warehouse engine is created lazily and separately.
        assert store.get_sql_engine() is not store._engine


# --------------------------------------------------------------------------
# 5.3 / 5.4 List reads
# --------------------------------------------------------------------------
class TestListReads:
    def test_database_table_column_lists(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        assert db_store.get_database_list() == ["sales"]
        assert db_store.get_table_list() == ["sales.Customers"]
        assert db_store.get_table_list(database="sales") == ["sales.Customers"]
        assert db_store.get_table_list(database="other") == []

        columns = db_store.get_column_list("Customers", database="sales")
        names = [c["column_name"] for c in columns]
        assert names == ["customer_id", "special_note"]

    def test_empty_database_name_no_leading_dot(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="")
        assert db_store.get_table_list() == ["Customers"]
        columns = db_store.get_column_list("Customers")
        assert [c["column_name"] for c in columns] == ["customer_id", "special_note"]

    def test_get_common_columns_without_table(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        common = db_store.get_column_list()
        # Only the common column definition should be returned.
        assert [c["column_name"] for c in common] == ["customer_id"]
        assert common[0]["is_common"] is True


# --------------------------------------------------------------------------
# 5.5 Common column membership
# --------------------------------------------------------------------------
class TestCommonColumnMembership:
    def test_common_column_reused_across_tables(self, db_store):
        common_col = {
            "column_name": "created_at",
            "type": "timestamp",
            "description": "Creation timestamp",
            "is_common": True,
        }
        db_store.save_table_information("Orders", _sample_information(), [common_col], database="sales")
        db_store.save_table_information("Invoices", _sample_information(), [common_col], database="sales")

        # Both tables should expose the common column.
        orders_cols = [c["column_name"] for c in db_store.get_column_list("Orders", "sales")]
        invoices_cols = [c["column_name"] for c in db_store.get_column_list("Invoices", "sales")]
        assert orders_cols == ["created_at"]
        assert invoices_cols == ["created_at"]

        # Only one reusable common column definition should exist.
        assert len(db_store.get_column_list()) == 1

    def test_is_common_derived_not_stored(self, db_store):
        # The ORM model must not carry a standalone is_common column; the flag is
        # derived from table scoping (scope_table_id).
        from openchatbi.catalog.store.database import CatalogColumn

        assert "is_common" not in CatalogColumn.__table__.columns
        assert "scope_table_id" in CatalogColumn.__table__.columns

        db_store.save_table_information(
            "Customers",
            {"description": "d"},
            [
                {"column_name": "shared", "type": "int", "is_common": True},
                {"column_name": "local", "type": "int", "is_common": False},
            ],
            database="sales",
        )
        cols = {c["column_name"]: c["is_common"] for c in db_store.get_column_list("Customers", "sales")}
        assert cols == {"shared": True, "local": False}

    def test_get_column_list_returns_only_owned_columns(self, db_store):
        db_store.save_table_information(
            "Customers",
            _sample_information(),
            [{"column_name": "cust_only", "type": "int", "is_common": False}],
            database="sales",
        )
        db_store.save_table_information(
            "Orders",
            _sample_information(),
            [{"column_name": "order_only", "type": "int", "is_common": False}],
            database="sales",
        )
        assert [c["column_name"] for c in db_store.get_column_list("Customers", "sales")] == ["cust_only"]
        assert [c["column_name"] for c in db_store.get_column_list("Orders", "sales")] == ["order_only"]


# --------------------------------------------------------------------------
# 5.6 Save/update table information
# --------------------------------------------------------------------------
class TestSaveTableInformation:
    def test_save_and_retrieve_all_keys(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        info = db_store.get_table_information("Customers", "sales")
        for key in ("description", "selection_rule", "sql_rule", "derived_metric"):
            assert key in info
        assert info["derived_metric"] == "count(distinct customer_id) as customer_count"

    def test_default_empty_string_keys(self, db_store):
        db_store.save_table_information(
            "Bare", {"description": "only description"}, [{"column_name": "x", "type": "int"}], database="sales"
        )
        info = db_store.get_table_information("Bare", "sales")
        assert info["description"] == "only description"
        assert info["selection_rule"] == ""
        assert info["sql_rule"] == ""
        assert info["derived_metric"] == ""

    def test_update_existing_false_keeps_metadata(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        db_store.save_table_information(
            "Customers",
            {"description": "CHANGED", "sql_rule": "CHANGED"},
            _sample_columns(),
            database="sales",
            update_existing=False,
        )
        info = db_store.get_table_information("Customers", "sales")
        assert info["description"] == "Customer information table"

    def test_update_existing_true_updates_metadata(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        db_store.save_table_information(
            "Customers",
            {"description": "CHANGED", "sql_rule": "NEW RULE"},
            _sample_columns(),
            database="sales",
            update_existing=True,
        )
        info = db_store.get_table_information("Customers", "sales")
        assert info["description"] == "CHANGED"
        assert info["sql_rule"] == "NEW RULE"

    def test_get_table_information_missing_returns_empty(self, db_store):
        assert db_store.get_table_information("Nope", "sales") == {}


# --------------------------------------------------------------------------
# 5.7 SQL examples
# --------------------------------------------------------------------------
class TestSqlExamples:
    def test_save_and_get_sql_examples(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        examples = [
            {"question": "How many customers?", "answer": "SELECT COUNT(*) FROM Customers"},
            {"question": "List customers", "answer": "SELECT * FROM Customers"},
        ]
        assert db_store.save_table_sql_examples("Customers", examples, database="sales") is True

        got = db_store.get_sql_examples("Customers", "sales")
        assert len(got) == 2
        assert got[0] == ("How many customers?", "SELECT COUNT(*) FROM Customers", ["sales.Customers"])

    def test_save_overwrites_examples(self, db_store):
        db_store.save_table_sql_examples("Customers", [{"question": "q1", "answer": "a1"}], database="sales")
        db_store.save_table_sql_examples("Customers", [{"question": "q2", "answer": "a2"}], database="sales")
        got = db_store.get_sql_examples("Customers", "sales")
        assert [q for q, _, _ in got] == ["q2"]

    def test_append_deduplicates_on_question(self, db_store):
        db_store.append_sql_example("q1", "SELECT 1", ["sales.Customers"])
        db_store.append_sql_example("q1", "SELECT 2", ["sales.Customers"])
        db_store.append_sql_example("q2", "SELECT 3", ["sales.Customers"])
        got = db_store.get_sql_examples("Customers", "sales")
        assert [q for q, _, _ in got] == ["q1", "q2"]

    def test_append_with_empty_tables_returns_false(self, db_store):
        assert db_store.append_sql_example("q1", "SELECT 1", []) is False

    def test_get_all_sql_examples(self, db_store):
        db_store.save_table_sql_examples("Customers", [{"question": "q1", "answer": "a1"}], database="sales")
        db_store.save_table_sql_examples("Orders", [{"question": "q2", "answer": "a2"}], database="sales")
        got = db_store.get_sql_examples()
        assert len(got) == 2


# --------------------------------------------------------------------------
# 5.8 Table selection examples
# --------------------------------------------------------------------------
class TestTableSelectionExamples:
    def test_save_and_get_selection_examples(self, db_store):
        examples = [
            ("Show me customers", ["Customers"]),
            ("Show customer orders", ["Customers", "Orders"]),
        ]
        assert db_store.save_table_selection_examples(examples) is True
        got = db_store.get_table_selection_examples()
        assert got == examples

    def test_save_overwrites_selection_examples(self, db_store):
        db_store.save_table_selection_examples([("q1", ["A"])])
        db_store.save_table_selection_examples([("q2", ["B"])])
        assert db_store.get_table_selection_examples() == [("q2", ["B"])]


# --------------------------------------------------------------------------
# 5.9 Existence check
# --------------------------------------------------------------------------
class TestCheckExists:
    def test_empty_store_returns_false(self, db_store):
        assert db_store.check_exists() is False

    def test_populated_store_returns_true(self, db_store):
        db_store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        assert db_store.check_exists() is True

    def test_table_without_columns_returns_true(self, db_store):
        # check_exists should look for tables, not table columns.
        db_store.save_table_information("Customers", _sample_information(), [], database="sales")
        assert db_store.check_exists() is True


# --------------------------------------------------------------------------
# 5.10 Factory integration
# --------------------------------------------------------------------------
class TestFactoryIntegration:
    def test_create_from_connection_string(self):
        store = create_catalog_store(
            "database", auto_load=False, connection_string="sqlite:///:memory:", auto_create_schema=True
        )
        assert isinstance(store, DatabaseCatalogStore)

    def test_create_from_engine(self):
        engine = create_engine("sqlite:///:memory:")
        store = create_catalog_store("database", auto_load=False, engine=engine, auto_create_schema=True)
        assert isinstance(store, DatabaseCatalogStore)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            create_catalog_store("unknown", auto_load=False)


# --------------------------------------------------------------------------
# 5.11 Equivalence with FileSystemCatalogStore
# --------------------------------------------------------------------------
class TestEquivalenceWithFileSystem:
    _SEMANTIC_FIELDS = ["column_name", "display_name", "type", "category", "description"]

    def _semantic_columns(self, columns):
        return [{k: c.get(k, "") for k in self._SEMANTIC_FIELDS} for c in columns]

    @pytest.fixture
    def fs_store(self, temp_dir):
        data_dir = temp_dir / "fs_catalog"
        data_dir.mkdir()
        return FileSystemCatalogStore(
            data_path=str(data_dir),
            data_warehouse_config={"uri": "sqlite:///:memory:", "database_name": "sales"},
        )

    def _populate(self, store):
        store.save_table_information("Customers", _sample_information(), _sample_columns(), database="sales")
        store.save_table_sql_examples("Customers", [{"question": "q1", "answer": "SELECT 1"}], database="sales")
        store.save_table_selection_examples([("Show customers", ["Customers"])])

    def test_equivalent_table_and_column_reads(self, db_store, fs_store):
        self._populate(db_store)
        self._populate(fs_store)

        assert db_store.get_database_list() == fs_store.get_database_list()
        assert db_store.get_table_list() == fs_store.get_table_list()

        db_cols = self._semantic_columns(db_store.get_column_list("Customers", "sales"))
        fs_cols = self._semantic_columns(fs_store.get_column_list("Customers", "sales"))
        assert db_cols == fs_cols

    def test_equivalent_table_information(self, db_store, fs_store):
        self._populate(db_store)
        self._populate(fs_store)
        keys = ["description", "selection_rule", "sql_rule"]
        db_info = db_store.get_table_information("Customers", "sales")
        fs_info = fs_store.get_table_information("Customers", "sales")
        assert {k: db_info.get(k, "") for k in keys} == {k: fs_info.get(k, "") for k in keys}

    def test_equivalent_sql_examples(self, db_store, fs_store):
        self._populate(db_store)
        self._populate(fs_store)
        assert db_store.get_sql_examples("Customers", "sales") == fs_store.get_sql_examples("Customers", "sales")

    def test_equivalent_selection_examples(self, db_store, fs_store):
        self._populate(db_store)
        self._populate(fs_store)
        assert db_store.get_table_selection_examples() == fs_store.get_table_selection_examples()
