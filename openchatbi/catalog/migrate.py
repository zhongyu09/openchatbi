"""Catalog migration utilities.

Provides a simple, backend-agnostic migration helper to copy catalog data from
one :class:`~openchatbi.catalog.catalog_store.CatalogStore` to another. The most
common use case is migrating an existing file-system catalog into a
database-backed catalog store.

Because both backends implement the same ``CatalogStore`` interface, migration is
just "read from source, write to destination" over the public methods.

CLI usage::

    python -m openchatbi.catalog.migrate \
        --source-path ./example \
        --dest-connection-string sqlite:///./catalog.db

Add ``--dry-run`` to preview without writing, and ``--no-update-existing`` to skip
overwriting rows that already exist in the destination.
"""

import argparse
import logging
from typing import Any

from openchatbi.catalog.catalog_store import CatalogStore, split_db_table_name

logger = logging.getLogger(__name__)


def migrate_catalog(
    source: CatalogStore,
    dest: CatalogStore,
    *,
    update_existing: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy all catalog data from ``source`` to ``dest``.

    Copies table metadata + columns, per-table SQL examples, and the global
    table-selection examples.

    Args:
        source: The catalog store to read from.
        dest: The catalog store to write to.
        update_existing: When ``True`` (default), overwrite table metadata/columns
            that already exist in the destination.
        dry_run: When ``True``, only log the planned operations without writing.

    Returns:
        A stats dict with counts: ``tables``, ``failed_tables``,
        ``sql_example_tables`` and ``selection_examples``.
    """
    stats: dict[str, int] = {
        "tables": 0,
        "failed_tables": 0,
        "sql_example_tables": 0,
        "selection_examples": 0,
    }

    full_table_names = source.get_table_list()
    logger.info("Found %d table(s) to migrate%s", len(full_table_names), " (dry-run)" if dry_run else "")

    for full_table_name in full_table_names:
        _, db_name, table_name = split_db_table_name(full_table_name)
        information: dict[str, Any] = source.get_table_information(full_table_name)
        columns: list[dict[str, Any]] = source.get_column_list(full_table_name)

        logger.info(
            "Migrating table '%s' (%d column(s))%s",
            full_table_name,
            len(columns),
            " [dry-run]" if dry_run else "",
        )

        if not dry_run:
            ok = dest.save_table_information(
                table_name,
                information,
                columns,
                database=db_name,
                update_existing=update_existing,
            )
            if not ok:
                logger.error("Failed to migrate table '%s'", full_table_name)
                stats["failed_tables"] += 1
                continue
        stats["tables"] += 1

        # Per-table SQL examples.
        examples = source.get_sql_examples(full_table_name)
        if examples:
            sql_examples = [{"question": question, "answer": answer} for question, answer, _ in examples]
            logger.info("  -> %d SQL example(s)", len(sql_examples))
            if not dry_run:
                dest.save_table_sql_examples(table_name, sql_examples, database=db_name)
            stats["sql_example_tables"] += 1

    # Global table-selection examples. Drop empty placeholder rows.
    selection_examples = [(q, tables) for q, tables in source.get_table_selection_examples() if q]
    if selection_examples:
        logger.info("Migrating %d table-selection example(s)", len(selection_examples))
        if not dry_run:
            dest.save_table_selection_examples(selection_examples)
        stats["selection_examples"] = len(selection_examples)

    logger.info("Migration complete: %s", stats)
    return stats


def _build_source(args: argparse.Namespace) -> CatalogStore:
    from openchatbi.catalog.store.file_system import FileSystemCatalogStore

    return FileSystemCatalogStore(data_path=args.source_path)


def _build_dest(args: argparse.Namespace) -> CatalogStore:
    from openchatbi.catalog.store.database import DatabaseCatalogStore

    return DatabaseCatalogStore(
        connection_string=args.dest_connection_string,
        auto_create_schema=True,
        echo=args.echo,
    )


def main(argv: list[str] | None = None) -> dict[str, int]:
    """CLI entry point: migrate a file-system catalog into a database catalog."""
    parser = argparse.ArgumentParser(description="Migrate a file-system catalog store into a database catalog store.")
    parser.add_argument(
        "--source-path",
        required=True,
        help="Path to the file-system catalog directory (e.g. ./example).",
    )
    parser.add_argument(
        "--dest-connection-string",
        required=True,
        help="SQLAlchemy URI for the destination catalog DB (e.g. sqlite:///./catalog.db).",
    )
    parser.add_argument(
        "--no-update-existing",
        action="store_true",
        help="Do not overwrite table metadata/columns that already exist in the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the migration without writing to the destination.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy engine echo for the destination DB (debugging).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    source = _build_source(args)
    dest = _build_dest(args)
    stats = migrate_catalog(
        source,
        dest,
        update_existing=not args.no_update_existing,
        dry_run=args.dry_run,
    )
    print(
        "Migration {}finished: {} tables, {} failed, {} tables with SQL examples, {} selection examples".format(
            "(dry-run) " if args.dry_run else "",
            stats["tables"],
            stats["failed_tables"],
            stats["sql_example_tables"],
            stats["selection_examples"],
        )
    )
    return stats


if __name__ == "__main__":
    main()
