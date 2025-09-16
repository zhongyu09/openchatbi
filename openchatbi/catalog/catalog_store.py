from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import Engine


class CatalogStore(ABC):
    """
    Abstract base class defining the storage interface for data catalog (database, table, column definitions, descriptions, and additional prompts).

    Common columns which have same meanings across tables will be store centralized to avoid data duplication.

    Column attribute:

        - column_name: the name of the column
        - display_name: the display name of the column
        - type: the data type of the column
        - category: dimension or metric
        - description: the description of the column
        - is_common: is common column or not
    """

    @abstractmethod
    def get_data_warehouse_config(self) -> dict:
        """
        Get the data warehouse configuration

        Returns:
            dict: Data warehouse configuration
        """
        pass

    @abstractmethod
    def get_sql_engine(self) -> Engine:
        """
        Get the SQLAlchemy engine for the catalog

        Returns:
            Engine: SQLAlchemy engine
        """
        pass

    @abstractmethod
    def get_database_list(self) -> list[str]:
        """
        Get a list of all databases

        Returns:
            List[str]: List of database names
        """
        pass

    @abstractmethod
    def get_table_list(self, database: str | None = None) -> list[str]:
        """
        Get a list of all tables in the specified database, if database is None, return all tables

        Args:
            database (Optional[str]): Database name

        Returns:
            List[str]: List of table names
        """
        pass

    @abstractmethod
    def get_column_list(self, table: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
        """
        Get all column information for the specified table, if table is None, return all common columns in the catalog

        Args:
            table (Optional[str]): Table name
            database (Optional[str]): Database name

        Returns:
            List[Dict[str, Any]]: List of column information, each column contains name, type, description, etc.
        """
        pass

    @abstractmethod
    def get_table_information(self, table: str, database: str | None = None) -> dict[str, Any]:
        """
        Get the information for the specified table

        Args:
            table (str): Table name
            database (Optional[str]): Database name

        Returns:
            Dict[str, Any]: Table information, including description text, selection rules, etc.
        """
        pass

    @abstractmethod
    def get_sql_examples(
        self, table: str | None = None, database: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        """
        Get SQL examples

        Args:
            table (Optional[str]): Table name
            database (Optional[str]): Database name

        Returns:
            List[Tuple[str, str, List[str]]]: List of SQL examples, each example is a Tuple-3 as (question, SQL, full_table_names)
        """
        pass

    @abstractmethod
    def get_table_selection_examples(self) -> list[tuple[str, list[str]]]:
        """
        Get table selection examples

        Returns:
            List[Tuple[str, List[str]]]: List of table selection examples, each example is a Tuple-2 as (question, selected tables)
        """
        pass

    @abstractmethod
    def save_table_information(
        self,
        table: str,
        information: dict[str, Any],
        columns: list[dict[str, Any]],
        database: str | None = None,
        update_existing: bool = False,
    ) -> bool:
        """
        Save the information and columns for a table

        Args:
            table (str): Table name
            information (Dict[str, Any]): Table information
            columns (List[Dict[str, Any]]): List of column information, each column dict contains at lease
                column_name, type, category, description
            database (Optional[str]): Database name
            update_existing (bool): Update existing table and column information

        Returns:
            bool: Whether the save was successful
        """
        pass

    @abstractmethod
    def save_table_sql_examples(self, table: str, examples: list[dict[str, str]], database: str | None = None) -> bool:
        """
        Save SQL examples for a table

        Args:
            table (str): Table name
            examples (List[Dict[str, str]]): List of SQL examples
            database (Optional[str]): Database name

        Returns:
            bool: Whether the save was successful
        """
        pass

    @abstractmethod
    def save_table_selection_examples(self, examples: list[tuple[str, list[str]]]) -> bool:
        """
        Save table selection examples

        Args:
            examples (List[Tuple[str, List[str]]]): List of table selection examples

        Returns:
            bool: Whether the save was successful
        """
        pass

    @abstractmethod
    def check_exists(self) -> bool:
        """
        Check if the catalog store has existing data/content

        Returns:
            bool: True if catalog store has existing data, False if empty or missing essential files
        """
        pass


def split_db_table_name(table: str, database: str | None = None) -> tuple[str, str, str]:
    """
    Split full table name into db name and table name
    Args:
        table (str): if database is None, should be full table name like `db.table`, otherwise should be only table name
        database (Optional[str]): Database name
    Returns:
        Tuple[str, str, str]: full_table_name, db_name, table_name

    """
    full_table_name = table
    if database is not None and "." not in table:
        full_table_name = f"{database}.{table}"
    if "." in full_table_name:
        db_name, table_name = full_table_name.rsplit(".", 1)
    else:
        db_name = ""
        table_name = full_table_name
    return full_table_name, db_name, table_name
