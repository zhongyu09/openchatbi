"""Database-backed catalog store implementation.

This module provides :class:`DatabaseCatalogStore`, a SQLAlchemy 2.0 backed
implementation of :class:`~openchatbi.catalog.catalog_store.CatalogStore`.

Compared to :class:`~openchatbi.catalog.store.file_system.FileSystemCatalogStore`
(which persists catalog metadata as CSV/YAML files), this backend stores the
catalog (databases, tables, columns, SQL examples and table-selection examples)
inside a relational database, enabling concurrent access, transactions,
indexing and cross-instance sharing.

Supported connection styles:

- ``connection_string``: any SQLAlchemy database URI, e.g. ``sqlite:///catalog.db``,
  ``postgresql+psycopg://user:pass@host/db`` or ``mysql+pymysql://user:pass@host/db``.
- ``engine``: an externally created SQLAlchemy :class:`~sqlalchemy.Engine`
  instance, useful for connection-pool sharing and testing.

Important: the catalog persistence database is configured independently from the
``data_warehouse_config``. The ``data_warehouse_config`` is only used to build the
data warehouse execution engine returned by :meth:`get_sql_engine`; it is never
reused as the catalog DB connection.
"""

import logging
import traceback
from typing import Any

from sqlalchemy import (
    JSON,
    Engine,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from ..catalog_store import CatalogStore, split_db_table_name
from ..helper import create_sqlalchemy_engine_instance

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all catalog ORM models."""


class CatalogDatabase(Base):
    """Logical database (namespace) that groups catalog tables."""

    __tablename__ = "catalog_databases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    tables: Mapped[list["CatalogTable"]] = relationship(back_populates="database")


class CatalogTable(Base):
    """A catalog table with its descriptive metadata."""

    __tablename__ = "catalog_tables"
    __table_args__ = (UniqueConstraint("database_id", "table_name", name="uq_catalog_table_db_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    database_id: Mapped[int] = mapped_column(ForeignKey("catalog_databases.id"), index=True)
    table_name: Mapped[str] = mapped_column(String(255), index=True)

    description: Mapped[str] = mapped_column(Text, default="")
    selection_rule: Mapped[str] = mapped_column(Text, default="")
    sql_rule: Mapped[str] = mapped_column(Text, default="")
    derived_metric: Mapped[str] = mapped_column(Text, default="")
    # Preserve any additional table-information keys for round-trip fidelity.
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    database: Mapped["CatalogDatabase"] = relationship(back_populates="tables")
    sql_examples: Mapped[list["CatalogSqlExample"]] = relationship(back_populates="table", cascade="all, delete-orphan")


class CatalogColumn(Base):
    """A column definition.

    A column definition does not by itself mean "this table owns this column".
    Table-column membership is expressed via :class:`CatalogTableColumn`, so a
    single common column definition can be reused across multiple tables.

    Whether a column is *common* (shared globally) or *table-specific* is derived
    solely from ``scope_table_id``:

    - ``scope_table_id IS NULL``  -> common column (``is_common=True`` in the API).
    - ``scope_table_id = <id>``   -> table-specific column scoped to that table
      (``is_common=False``), so it is never accidentally reused by other tables.

    There is intentionally no stored ``is_common`` boolean: it would duplicate the
    information already encoded by ``scope_table_id`` and risk the two getting out
    of sync. The ``is_common`` flag is still accepted on input and returned on
    output for backward compatibility.
    """

    __tablename__ = "catalog_columns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    column_name: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(255), default="")
    alias: Mapped[str] = mapped_column(String(255), default="")
    type: Mapped[str] = mapped_column(String(255), default="")
    category: Mapped[str] = mapped_column(String(255), default="")
    tag: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    default: Mapped[str] = mapped_column("default_value", String(255), default="")
    # NULL => common (globally shared) column; set => table-specific column scoped
    # to the owning table. This single field encodes the common/table-specific split.
    scope_table_id: Mapped[int | None] = mapped_column(ForeignKey("catalog_tables.id"), nullable=True, index=True)
    # Preserve any additional column metadata keys for round-trip fidelity.
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class CatalogTableColumn(Base):
    """Explicit table-column membership (maps to the legacy ``table_columns.csv``)."""

    __tablename__ = "catalog_table_columns"
    __table_args__ = (UniqueConstraint("table_id", "column_id", name="uq_catalog_table_column"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("catalog_tables.id"), index=True)
    column_id: Mapped[int] = mapped_column(ForeignKey("catalog_columns.id"), index=True)


class CatalogSqlExample(Base):
    """A per-table natural-language question / SQL answer example."""

    __tablename__ = "catalog_sql_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("catalog_tables.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)

    table: Mapped["CatalogTable"] = relationship(back_populates="sql_examples")


class CatalogTableSelectionExample(Base):
    """A global table-selection example (question -> selected tables)."""

    __tablename__ = "catalog_table_selection_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    selected_tables: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)


# Column metadata fields modeled as first-class ORM columns.
_KNOWN_COLUMN_FIELDS = {
    "column_name",
    "display_name",
    "alias",
    "type",
    "category",
    "tag",
    "description",
    "default",
    "is_common",
}

# Table-information fields modeled as first-class ORM columns.
_KNOWN_TABLE_FIELDS = {"description", "selection_rule", "sql_rule", "derived_metric"}


class DatabaseCatalogStore(CatalogStore):
    """SQLAlchemy-backed catalog store.

    Args:
        connection_string (Optional[str]): SQLAlchemy database URI for the catalog
            persistence database. Mutually usable with ``engine`` (one is required).
        engine (Optional[Engine]): Pre-built SQLAlchemy engine for the catalog DB.
        data_warehouse_config (Optional[dict]): Data warehouse configuration used
            *only* to build the data warehouse execution engine returned by
            :meth:`get_sql_engine`. Never used as the catalog DB connection.
        auto_create_schema (bool): When ``True`` (default) run
            ``Base.metadata.create_all()`` on initialization so the required
            tables are created for development and testing convenience. Set to
            ``False`` for externally managed schemas.
        echo (bool): SQLAlchemy engine ``echo`` flag (only used when creating an
            engine from ``connection_string``).
    """

    def __init__(
        self,
        connection_string: str | None = None,
        engine: Engine | None = None,
        data_warehouse_config: dict[str, Any] | None = None,
        auto_create_schema: bool = True,
        echo: bool = False,
    ) -> None:
        if engine is None and (not connection_string or not connection_string.strip()):
            raise ValueError("Either 'engine' or a non-empty 'connection_string' must be provided")

        if data_warehouse_config is None:
            data_warehouse_config = {}
        elif not isinstance(data_warehouse_config, dict):
            raise ValueError("data_warehouse_config must be a dictionary")

        self._engine: Engine = engine if engine is not None else create_engine(connection_string, echo=echo)  # type: ignore[arg-type]
        self._session_factory = sessionmaker(bind=self._engine)

        self._data_warehouse_config = data_warehouse_config
        self._sql_engine: Engine | None = None

        if auto_create_schema:
            Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # Data warehouse access (independent of the catalog DB)
    # ------------------------------------------------------------------
    def get_data_warehouse_config(self) -> dict:
        return self._data_warehouse_config

    def get_sql_engine(self) -> Engine:
        if self._sql_engine is None:
            try:
                self._sql_engine = create_sqlalchemy_engine_instance(self._data_warehouse_config)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError("SQL engine is not available. Check data warehouse configuration.") from e
        return self._sql_engine

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _full_name(db_name: str, table_name: str) -> str:
        """Build a full table name, avoiding a leading dot for empty databases."""
        return f"{db_name}.{table_name}" if db_name else table_name

    @staticmethod
    def _validate_table_name(table: str) -> None:
        if not table or not isinstance(table, str):
            raise ValueError("Table name must be a non-empty string")
        invalid_chars = ["/", "\\", "*", "?", "<", ">", "|", '"', "'"]
        if any(char in table for char in invalid_chars):
            raise ValueError(f"Table name contains invalid characters: {table}")

    @staticmethod
    def _validate_sql_examples(examples: list[dict[str, str]]) -> None:
        if not isinstance(examples, list):
            raise ValueError("Examples must be a list")
        required_fields = {"question", "answer"}
        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"Example {i} must be a dictionary")
            missing_fields = required_fields - set(example.keys())
            if missing_fields:
                raise ValueError(f"Example {i} missing required fields: {missing_fields}")
            for field in required_fields:
                value = example.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Example {i}: {field} must be a non-empty string")

    def _get_database(self, session: Session, db_name: str) -> CatalogDatabase | None:
        return session.scalars(select(CatalogDatabase).where(CatalogDatabase.name == db_name)).first()

    def _get_or_create_database(self, session: Session, db_name: str) -> CatalogDatabase:
        db_row = self._get_database(session, db_name)
        if db_row is None:
            db_row = CatalogDatabase(name=db_name)
            session.add(db_row)
            session.flush()
        return db_row

    def _get_table_row(self, session: Session, table: str, database: str | None) -> CatalogTable | None:
        _, db_name, table_name = split_db_table_name(table, database)
        stmt = (
            select(CatalogTable)
            .join(CatalogDatabase, CatalogTable.database_id == CatalogDatabase.id)
            .where(CatalogDatabase.name == db_name, CatalogTable.table_name == table_name)
        )
        return session.scalars(stmt).first()

    def _get_or_create_table(self, session: Session, db_name: str, table_name: str) -> CatalogTable:
        db_row = self._get_or_create_database(session, db_name)
        session.flush()
        row = session.scalars(
            select(CatalogTable).where(CatalogTable.database_id == db_row.id, CatalogTable.table_name == table_name)
        ).first()
        if row is None:
            row = CatalogTable(database_id=db_row.id, table_name=table_name)
            session.add(row)
            session.flush()
        return row

    @staticmethod
    def _column_to_dict(col: CatalogColumn) -> dict[str, Any]:
        data: dict[str, Any] = {
            "column_name": col.column_name,
            "display_name": col.display_name,
            "alias": col.alias,
            "type": col.type,
            "category": col.category,
            "tag": col.tag,
            "description": col.description,
            "default": col.default,
        }
        if col.extra:
            for key, value in col.extra.items():
                data.setdefault(key, value)
        # Common vs table-specific is derived from scope_table_id.
        data["is_common"] = col.scope_table_id is None
        return data

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_database_list(self) -> list[str]:
        with self._session_factory() as session:
            stmt = (
                select(CatalogDatabase.name)
                .join(CatalogTable, CatalogTable.database_id == CatalogDatabase.id)
                .distinct()
            )
            return list(session.scalars(stmt).all())

    def get_table_list(self, database: str | None = None) -> list[str]:
        with self._session_factory() as session:
            stmt = select(CatalogDatabase.name, CatalogTable.table_name).join(
                CatalogDatabase, CatalogTable.database_id == CatalogDatabase.id
            )
            if database is not None:
                stmt = stmt.where(CatalogDatabase.name == database)
            stmt = stmt.order_by(CatalogTable.id)
            rows = session.execute(stmt).all()
            return [self._full_name(db_name, table_name) for db_name, table_name in rows]

    def get_column_list(self, table: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            if table is None:
                # Common columns are those not scoped to any specific table.
                stmt = select(CatalogColumn).where(CatalogColumn.scope_table_id.is_(None)).order_by(CatalogColumn.id)
                return [self._column_to_dict(col) for col in session.scalars(stmt).all()]

            table_row = self._get_table_row(session, table, database)
            if table_row is None:
                return []

            stmt = (
                select(CatalogColumn)
                .join(CatalogTableColumn, CatalogTableColumn.column_id == CatalogColumn.id)
                .where(CatalogTableColumn.table_id == table_row.id)
                .order_by(CatalogTableColumn.id)
            )
            return [self._column_to_dict(col) for col in session.scalars(stmt).all()]

    def get_table_information(self, table: str, database: str | None = None) -> dict[str, Any]:
        with self._session_factory() as session:
            table_row = self._get_table_row(session, table, database)
            if table_row is None:
                return {}
            info: dict[str, Any] = {
                "description": table_row.description or "",
                "selection_rule": table_row.selection_rule or "",
                "sql_rule": table_row.sql_rule or "",
                "derived_metric": table_row.derived_metric or "",
            }
            if table_row.extra:
                for key, value in table_row.extra.items():
                    info.setdefault(key, value)
            return info

    def get_sql_examples(
        self, table: str | None = None, database: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        with self._session_factory() as session:
            if table is None:
                stmt = (
                    select(
                        CatalogDatabase.name,
                        CatalogTable.table_name,
                        CatalogSqlExample.question,
                        CatalogSqlExample.answer,
                    )
                    .join(CatalogTable, CatalogSqlExample.table_id == CatalogTable.id)
                    .join(CatalogDatabase, CatalogTable.database_id == CatalogDatabase.id)
                    .order_by(CatalogSqlExample.id)
                )
                rows = session.execute(stmt).all()
                return [
                    (question, answer, [self._full_name(db_name, table_name)])
                    for db_name, table_name, question, answer in rows
                ]

            table_row = self._get_table_row(session, table, database)
            if table_row is None:
                return []
            full_table_name = self._full_name(table_row.database.name, table_row.table_name)
            table_stmt = (
                select(CatalogSqlExample.question, CatalogSqlExample.answer)
                .where(CatalogSqlExample.table_id == table_row.id)
                .order_by(CatalogSqlExample.id)
            )
            table_rows = session.execute(table_stmt).all()
            return [(question, answer, [full_table_name]) for question, answer in table_rows]

    def get_table_selection_examples(self) -> list[tuple[str, list[str]]]:
        with self._session_factory() as session:
            stmt = select(CatalogTableSelectionExample).order_by(CatalogTableSelectionExample.id)
            return [(row.question, list(row.selected_tables or [])) for row in session.scalars(stmt).all()]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def _find_column(self, session: Session, column_name: str, is_common: bool, table_id: int) -> CatalogColumn | None:
        stmt = select(CatalogColumn).where(CatalogColumn.column_name == column_name)
        if is_common:
            # Common columns are global: not scoped to any table.
            stmt = stmt.where(CatalogColumn.scope_table_id.is_(None))
        else:
            # Table-specific columns are scoped to the owning table.
            stmt = stmt.where(CatalogColumn.scope_table_id == table_id)
        return session.scalars(stmt).first()

    @staticmethod
    def _build_column_kwargs(column: dict[str, Any]) -> dict[str, Any]:
        extra = {k: str(v) for k, v in column.items() if k not in _KNOWN_COLUMN_FIELDS}
        return {
            "display_name": str(column.get("display_name", "") or ""),
            "alias": str(column.get("alias", "") or ""),
            "type": str(column.get("type", "") or ""),
            "category": str(column.get("category", "") or ""),
            "tag": str(column.get("tag", "") or ""),
            "description": str(column.get("description", "") or ""),
            "default": str(column.get("default", "") or ""),
            "extra": extra or None,
        }

    def _ensure_membership(self, session: Session, table_id: int, column_id: int) -> None:
        exists = session.scalars(
            select(CatalogTableColumn).where(
                CatalogTableColumn.table_id == table_id, CatalogTableColumn.column_id == column_id
            )
        ).first()
        if exists is None:
            session.add(CatalogTableColumn(table_id=table_id, column_id=column_id))

    def _save_columns(
        self, session: Session, table_row: CatalogTable, columns: list[dict[str, Any]], update_existing: bool
    ) -> None:
        for column in columns:
            if "column_name" not in column:
                continue
            column_name = column["column_name"]
            is_common = bool(column.get("is_common", False))

            col_row = self._find_column(session, column_name, is_common, table_row.id)
            if col_row is None:
                col_row = CatalogColumn(
                    column_name=column_name,
                    scope_table_id=None if is_common else table_row.id,
                    **self._build_column_kwargs(column),
                )
                session.add(col_row)
                session.flush()
            elif update_existing:
                for key, value in self._build_column_kwargs(column).items():
                    setattr(col_row, key, value)

            self._ensure_membership(session, table_row.id, col_row.id)

    def save_table_information(
        self,
        table: str,
        information: dict[str, Any],
        columns: list[dict[str, Any]],
        database: str | None = None,
        update_existing: bool = False,
    ) -> bool:
        # Validation errors propagate (mirrors FileSystemCatalogStore behavior).
        self._validate_table_name(table)
        if not isinstance(information, dict):
            raise ValueError("Table information must be a dictionary")
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list")

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)
            with self._session_factory() as session:
                db_row = self._get_or_create_database(session, db_name)
                session.flush()

                table_row = session.scalars(
                    select(CatalogTable).where(
                        CatalogTable.database_id == db_row.id, CatalogTable.table_name == table_name
                    )
                ).first()

                extra = {k: v for k, v in information.items() if k not in _KNOWN_TABLE_FIELDS}
                if table_row is None:
                    table_row = CatalogTable(
                        database_id=db_row.id,
                        table_name=table_name,
                        description=str(information.get("description", "") or ""),
                        selection_rule=str(information.get("selection_rule", "") or ""),
                        sql_rule=str(information.get("sql_rule", "") or ""),
                        derived_metric=str(information.get("derived_metric", "") or ""),
                        extra=extra or None,
                    )
                    session.add(table_row)
                    session.flush()
                elif update_existing:
                    table_row.description = str(information.get("description", "") or "")
                    table_row.selection_rule = str(information.get("selection_rule", "") or "")
                    table_row.sql_rule = str(information.get("sql_rule", "") or "")
                    table_row.derived_metric = str(information.get("derived_metric", "") or "")
                    table_row.extra = extra or None

                self._save_columns(session, table_row, columns, update_existing)
                session.commit()

            logger.info(f"Successfully saved table information for {full_table_name}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error when saving table information: {e}")
            logger.error(traceback.format_stack())
            return False

    def save_table_sql_examples(self, table: str, examples: list[dict[str, str]], database: str | None = None) -> bool:
        self._validate_table_name(table)
        self._validate_sql_examples(examples)

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)
            with self._session_factory() as session:
                table_row = self._get_or_create_table(session, db_name, table_name)
                # Overwrite the per-table example block.
                session.execute(delete(CatalogSqlExample).where(CatalogSqlExample.table_id == table_row.id))
                for example in examples:
                    session.add(
                        CatalogSqlExample(table_id=table_row.id, question=example["question"], answer=example["answer"])
                    )
                session.commit()

            logger.info(f"Successfully saved {len(examples)} examples for table {full_table_name}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error when saving table examples: {e}")
            logger.error(traceback.format_stack())
            return False

    def append_sql_example(
        self,
        question: str,
        sql: str,
        tables: list[str],
        source: str = "golden",
        database: str | None = None,
    ) -> bool:
        self._validate_sql_examples([{"question": question, "answer": sql}])
        if not tables:
            logger.warning("No target tables provided for SQL example; skipping append.")
            return False
        try:
            target_table = tables[0]
            full_table_name, db_name, table_name = split_db_table_name(target_table, database)
            with self._session_factory() as session:
                table_row = self._get_or_create_table(session, db_name, table_name)

                existing = session.scalars(
                    select(CatalogSqlExample).where(
                        CatalogSqlExample.table_id == table_row.id, CatalogSqlExample.question == question
                    )
                ).first()
                if existing is not None:
                    logger.info(f"Golden SQL example already present for table {full_table_name}; skipping append.")
                    return True

                session.add(CatalogSqlExample(table_id=table_row.id, question=question, answer=sql))
                session.commit()

            logger.info(f"Appended {source} SQL example for table {full_table_name}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error when appending SQL example: {e}")
            logger.error(traceback.format_stack())
            return False

    def save_table_selection_examples(self, examples: list[tuple[str, list[str]]]) -> bool:
        try:
            with self._session_factory() as session:
                # Overwrite all existing selection examples.
                session.execute(delete(CatalogTableSelectionExample))
                for question, selected_tables in examples:
                    session.add(CatalogTableSelectionExample(question=question, selected_tables=list(selected_tables)))
                session.commit()

            logger.info(f"Successfully saved {len(examples)} table selection examples.")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error when saving table selection examples: {e}")
            logger.error(traceback.format_stack())
            return False

    def check_exists(self) -> bool:
        try:
            with self._session_factory() as session:
                return session.scalars(select(CatalogTable.id).limit(1)).first() is not None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error checking catalog existence: {e}")
            return False
