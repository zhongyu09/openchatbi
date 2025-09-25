"""File system-based catalog store implementation."""

import csv
import logging
import os
import re
import traceback
from typing import Any

import yaml
from sqlalchemy import Engine

from ..catalog_store import CatalogStore, split_db_table_name
from ..helper import create_sqlalchemy_engine_instance

logger = logging.getLogger(__name__)


class FileSystemCatalogStore(CatalogStore):
    """File system-based data catalog storage implementation.

    Stores catalog data in CSV and YAML files on the local filesystem.
    """

    data_path: str
    table_info_file: str
    sql_example_file: str
    table_selection_example_file: str
    table_columns_file: str
    common_columns_file: str
    table_spec_columns_file: str

    _table_info_cache: dict | None
    _table_columns_cache: dict | None
    _common_columns_cache: dict | None
    _table_spec_columns_cache: dict | None
    _sql_example_cache: dict | None
    _table_selection_example_cache: dict | None

    _data_warehouse_config: dict
    _sql_engine: Engine

    def __init__(self, data_path: str, data_warehouse_config: dict):
        """Initialize filesystem catalog store.

        Args:
            data_path (str): Directory absolute path for storing catalog files.
            data_warehouse_config (dict): Data warehouse configuration dictionary with keys:
                - uri (str): Database connection URI
                - include_tables (Optional[List[str]]): List of tables to include, if None include all
                - database_name (Optional[str]): Database name to use in catalog
        """
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError("data_path must be a non-empty string")

        if data_warehouse_config is None:
            data_warehouse_config = {}
        elif not isinstance(data_warehouse_config, dict):
            raise ValueError("data_warehouse_config must be a dictionary")

        self.data_path = data_path.strip()
        self.table_info_file = os.path.join(data_path, "table_info.yaml")
        self.sql_example_file = os.path.join(data_path, "sql_example.yaml")
        self.table_selection_example_file = os.path.join(data_path, "table_selection_example.csv")
        self.table_columns_file = os.path.join(data_path, "table_columns.csv")
        self.common_columns_file = os.path.join(data_path, "common_columns.csv")
        self.table_spec_columns_file = os.path.join(data_path, "table_spec_columns.csv")

        # Ensure directory exists with proper error handling
        try:
            os.makedirs(self.data_path, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to create data directory '{self.data_path}': {e}") from e

        # Initialize cache
        self._table_info_cache = None
        self._table_columns_cache = None
        self._common_columns_cache = None
        self._table_spec_columns_cache = None
        self._sql_example_cache = None
        self._table_selection_example_cache = None

        self._data_warehouse_config = data_warehouse_config
        try:
            self._sql_engine = create_sqlalchemy_engine_instance(data_warehouse_config)
        except Exception as e:
            logger.warning(f"Failed to create SQL engine: {e}. Some catalog operations may not work.")
            self._sql_engine = None

    def _clear_cache(self) -> None:
        """
        Clear all cached data to ensure consistency after data modifications
        """
        self._table_info_cache = None
        self._table_columns_cache = None
        self._common_columns_cache = None
        self._table_spec_columns_cache = None
        self._sql_example_cache = None
        self._table_selection_example_cache = None
        logger.debug("Cleared all caches")

    def get_data_warehouse_config(self) -> dict:
        return self._data_warehouse_config

    def get_sql_engine(self) -> Engine:
        if self._sql_engine is None:
            raise RuntimeError("SQL engine is not available. Check data warehouse configuration.")
        return self._sql_engine

    def _validate_table_name(self, table: str) -> bool:
        """
        Validate table name

        Args:
            table (str): Table name

        Returns:
            bool: Whether the table name is valid

        Raises:
            ValueError: If table name is invalid
        """
        if not table or not isinstance(table, str):
            raise ValueError("Table name must be a non-empty string")

        # Check for invalid characters (allow dots for db.table format)
        invalid_chars = ["/", "\\", "*", "?", "<", ">", "|", '"', "'"]
        if any(char in table for char in invalid_chars):
            raise ValueError(f"Table name contains invalid characters: {table}")

        return True

    def _validate_column_data(self, columns: list[dict[str, Any]]) -> bool:
        """
        Validate column data format

        Args:
            columns (List[Dict[str, Any]]): List of column information

        Returns:
            bool: Whether the column data is valid

        Raises:
            ValueError: If column data is invalid
        """
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list")

        required_fields = {"column_name", "type"}

        for i, column in enumerate(columns):
            if not isinstance(column, dict):
                raise ValueError(f"Column {i} must be a dictionary")

            # Check required fields
            missing_fields = required_fields - set(column.keys())
            if missing_fields:
                raise ValueError(f"Column {i} missing required fields: {missing_fields}")

            # Validate column_name
            column_name = column.get("column_name")
            if not isinstance(column_name, str) or not column_name.strip():
                raise ValueError(f"Column {i}: column_name must be a non-empty string")

            # Validate type
            column_type = column.get("type")
            if not isinstance(column_type, str) or not column_type.strip():
                raise ValueError(f"Column {i}: type must be a non-empty string")

        return True

    def _validate_table_information(self, information: dict[str, Any]) -> bool:
        """
        Validate table information format

        Args:
            information (Dict[str, Any]): Table information

        Returns:
            bool: Whether the table information is valid

        Raises:
            ValueError: If table information is invalid
        """
        if not isinstance(information, dict):
            raise ValueError("Table information must be a dictionary")

        # Validate optional string fields
        string_fields = ["description", "selection_rule"]
        for field in string_fields:
            if field in information:
                value = information[field]
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"Table information field '{field}' must be a string or None")

        return True

    def _validate_sql_examples(self, examples: list[dict[str, str]]) -> bool:
        """
        Validate SQL examples format

        Args:
            examples (List[Dict[str, str]]): List of SQL examples

        Returns:
            bool: Whether the SQL examples are valid

        Raises:
            ValueError: If SQL examples are invalid
        """
        if not isinstance(examples, list):
            raise ValueError("Examples must be a list")

        required_fields = {"question", "answer"}

        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"Example {i} must be a dictionary")

            # Check required fields
            missing_fields = required_fields - set(example.keys())
            if missing_fields:
                raise ValueError(f"Example {i} missing required fields: {missing_fields}")

            # Validate fields are non-empty strings
            for field in required_fields:
                value = example.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Example {i}: {field} must be a non-empty string")

        return True

    @staticmethod
    def _load_yaml_file(file_path: str) -> dict:
        """
        Load YAML file

        Args:
            file_path (str): File path

        Returns:
            Dict: YAML content
        """
        if not os.path.exists(file_path):
            logger.debug(f"YAML file does not exist: {file_path}")
            return {}

        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                logger.debug(f"Successfully loaded YAML file: {file_path}")
                return data
        except Exception as e:
            logger.error(f"Failed to load YAML file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return {}

    @staticmethod
    def _load_csv_file(file_path: str) -> list[dict[str, str]]:
        """
        Load CSV file

        Args:
            file_path (str): File path

        Returns:
            List[Dict[str, str]]: List of rows as dictionaries
        """
        if not os.path.exists(file_path):
            logger.debug(f"CSV file does not exist: {file_path}")
            return []

        try:
            result = []
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    result.append(row)
            logger.debug(f"Successfully loaded CSV file: {file_path} with {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return []

    @staticmethod
    def _save_yaml_file(file_path: str, data: dict) -> bool:
        """
        Save YAML file

        Args:
            file_path (str): File path
            data (Dict): Data to save

        Returns:
            bool: Whether the save was successful
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save YAML file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return False

    @staticmethod
    def _save_csv_file(file_path: str, data: list[dict[str, str]], headers: list[str] = None) -> bool:
        """
        Save CSV file

        Args:
            file_path (str): File path
            data (List[Dict[str, str]]): List of rows as dictionaries
            headers (List[str]): List of header names in sequence

        Returns:
            bool: Whether the save was successful
        """
        try:
            if not data:
                return True

            # Get all possible headers from all rows
            all_headers = set()
            for row in data:
                all_headers.update(row.keys())

            # If specify field_names, make sure all keys are in field_names
            if headers is not None:
                for key in all_headers:
                    if key not in headers:
                        headers.append(key)

            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)

            return True
        except Exception as e:
            logger.error(f"Failed to save CSV file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return False

    def _load_tables(self) -> dict[str, list[str]]:
        # Load table_columns.csv
        table_columns_csv = self._load_csv_file(self.table_columns_file)

        # Get unique db_name.table_name combinations
        table_dict = {}
        for row in table_columns_csv:
            if "db_name" in row and "table_name" in row and "column_name" in row:
                db_name = row["db_name"]
                table_name = row["table_name"]
                column_name = row["column_name"]
                full_table_name = f"{db_name}.{table_name}"
                if full_table_name not in table_dict:
                    table_dict[full_table_name] = []
                table_dict[full_table_name].append(column_name)
        return table_dict

    def _load_common_columns(self) -> dict[str, dict[str, Any]]:
        # Load common_columns.csv to get column details
        columns_csv = self._load_csv_file(self.common_columns_file)

        # Filter and return column details
        column_dict = {}
        for row in columns_csv:
            if row.get("column_name") and row.get("type"):
                # Convert row to Dict[str, Any]
                column_info = {}
                for key, value in row.items():
                    if key != "":
                        column_info[key] = value
                column_dict[row["column_name"]] = column_info

        return column_dict

    def _load_table_spec_columns(self) -> dict[str, dict[str, Any]]:
        """
        Load info of table spec columns
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of table specific columns information, keyed by "full_table_name:column_name"
        """
        # Load table_spec_columns.csv to get table specific column details
        columns_csv = self._load_csv_file(self.table_spec_columns_file)

        # Filter and return column details
        column_dict = {}
        for row in columns_csv:
            if "db_name" in row and "table_name" in row and "column_name" in row and row["column_name"]:
                # Convert row to Dict[(str, str), Any]
                full_table_name = f"{row['db_name']}.{row['table_name']}"
                column_info = {}
                for key, value in row.items():
                    if key != "":
                        column_info[key] = value
                column_dict[f"{full_table_name}:{row['column_name']}"] = column_info

        return column_dict

    def _parse_example_text(self, example_text: str) -> list[tuple[str, str]]:
        """
        Parse example text, format is Q: ... A: ...

        Args:
            example_text (str): Example text

        Returns:
            List[Tuple[str, str]]: List of parsed question-answer pairs
        """
        examples = []
        lines = example_text.strip().split("\n")

        question = ""
        answer = ""
        current_type = None

        for line in lines:
            if line.startswith("Q:"):
                # If there is already a complete question-answer pair, add it to the results
                if question and answer:
                    examples.append((question.strip(), answer.strip()))
                    question = ""
                    answer = ""

                question = line[2:]
                current_type = "Q"
            elif line.startswith("A:"):
                answer = line[2:]
                current_type = "A"
            else:
                # Continue adding to the current type
                if current_type == "Q":
                    question += "\n" + line
                elif current_type == "A":
                    answer += "\n" + line

        # Add the last question-answer pair
        if question and answer:
            examples.append((question.strip(), answer.strip()))

        return examples

    def get_database_list(self) -> list[str]:
        # Extract unique database names
        databases = set()
        for table in self._get_all_table_schema().keys():
            full_table_name, db_name, table_name = split_db_table_name(table)
            databases.add(db_name)

        return list(databases)

    def _get_all_table_schema(self) -> dict[str, list[str]]:
        """
        Get all tables schema (columns of table)
        Returns:
            Dict[str, List[str]]: Tables schema (columns) dict, keyed by table name
        """
        if self._table_columns_cache is None:
            self._table_columns_cache = self._load_tables()
        # Return a deep copy to prevent external modifications
        return {k: v.copy() for k, v in self._table_columns_cache.items()}

    def get_table_list(self, database: str | None = None) -> list[str]:
        tables = self._get_all_table_schema()
        if database is None:
            return list(tables.keys())

        # Filter by database
        filtered_tables = []
        for full_table_name in tables.keys():
            _, db_name, table_name = split_db_table_name(full_table_name)
            if db_name == database:
                filtered_tables.append(full_table_name)

        return filtered_tables

    def _get_common_columns(self) -> dict[str, dict[str, Any]]:
        """
        Get information of all common columns
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of columns information, keyed by column name
        """
        if self._common_columns_cache is None:
            self._common_columns_cache = self._load_common_columns()
        # Return a deep copy to prevent external modifications
        return {k: v.copy() for k, v in self._common_columns_cache.items()}

    def _get_table_spec_columns(self) -> dict[str, dict[str, Any]]:
        """
        Get information of all table specific columns
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of table specific columns information, keyed by "full_table_name:column_name"
        """
        if self._table_spec_columns_cache is None:
            self._table_spec_columns_cache = self._load_table_spec_columns()
        # Return a deep copy to prevent external modifications
        return {k: v.copy() for k, v in self._table_spec_columns_cache.items()}

    def get_column_list(self, table: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
        _common_columns = self._get_common_columns()
        if table is None:
            return list(_common_columns.values())

        # Get the full table name
        full_table_name, db_name, table_name = split_db_table_name(table, database)

        # Filter table columns
        tables_dict = self._get_all_table_schema()
        if full_table_name not in tables_dict:
            return []

        table_columns = tables_dict[full_table_name]

        # If no columns found, return empty list
        if not table_columns:
            return []

        # Filter and return column details
        result = []
        _table_spec_columns = self._get_table_spec_columns()
        for column in table_columns:
            # check if the column is table specific
            key = f"{full_table_name}:{column}"
            if key in _table_spec_columns:
                column_info = _table_spec_columns[key]
                column_info["is_common"] = False
                result.append(column_info)
            else:
                column_info = _common_columns.get(column)
                if column_info:
                    column_info["is_common"] = True
                    result.append(column_info)
        return result

    def get_table_information(self, table: str, database: str | None = None) -> dict[str, Any]:
        full_table_name, db_name, table_name = split_db_table_name(table, database)

        if self._table_info_cache is None:
            self._table_info_cache = self._load_yaml_file(self.table_info_file)

        if db_name in self._table_info_cache and table_name in self._table_info_cache[db_name]:
            # Return a copy to prevent external modifications
            return self._table_info_cache[db_name][table_name].copy()

        return {}

    def get_sql_examples(
        self, table: str | None = None, database: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        if self._sql_example_cache is None:
            self._sql_example_cache = self._load_yaml_file(self.sql_example_file)

        if table is None:
            # If no table specified, return all examples
            examples = []
            for db_name, tables in self._sql_example_cache.items():
                for table_name, example_text in tables.items():
                    qa_pairs = self._parse_example_text(example_text)
                    examples.extend([(q, a, [f"{db_name}.{table_name}"]) for (q, a) in qa_pairs])
            return examples

        full_table_name, db_name, table_name = split_db_table_name(table, database)

        # Find examples that include this table
        examples = []

        # Check the fact section
        if db_name in self._sql_example_cache:
            if table_name in self._sql_example_cache[db_name]:
                # Parse example text, format is Q: ... A: ...
                qa_pairs = self._parse_example_text(self._sql_example_cache[db_name][table_name])
                examples.extend([(q, a, [full_table_name]) for (q, a) in qa_pairs])

        return examples

    @staticmethod
    def _load_table_selection_examples_from_csv(file_path: str) -> list[tuple[str, list[str]]]:
        examples = []
        try:
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = row.get("question", "").strip()
                    selected_tables = row.get("selected_tables", "").strip()
                    if question and selected_tables:
                        table_list = [p.strip() for p in re.split(r"[ ,\n]", selected_tables) if p.strip()]
                        examples.append((question, table_list))
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to load table selection examples from {file_path}: {e}")
        return examples

    def get_table_selection_examples(self) -> list[tuple[str, list[str]]]:
        if self._table_selection_example_cache is None:
            self._table_selection_example_cache = self._load_table_selection_examples_from_csv(
                self.table_selection_example_file
            )
        return self._table_selection_example_cache

    def save_table_information(
        self,
        table: str,
        information: dict[str, Any],
        columns: list[dict[str, Any]],
        database: str | None = None,
        update_existing: bool = False,
    ) -> bool:
        # Validate input data (let validation errors propagate)
        self._validate_table_name(table)
        self._validate_table_information(information)
        self._validate_column_data(columns)

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)

            table_info = self._load_yaml_file(self.table_info_file)

            # Save columns first
            if not self._save_columns(table_name, columns, db_name, update_existing):
                logger.error(f"Failed to save columns for table {full_table_name}")
                return False

            # Save table information (ensure proper structure)
            if db_name not in table_info:
                table_info[db_name] = {}
            if update_existing or table_name not in table_info[db_name]:
                table_info[db_name][table_name] = information
            success = self._save_yaml_file(self.table_info_file, table_info)

            if success:
                logger.info(f"Successfully saved table information for {full_table_name}")
                # Clear cache to ensure consistency
                self._clear_cache()

            return success
        except Exception as e:
            logger.error(f"Unexpected error when saving table information: {e}")
            logger.error(traceback.format_stack())
            return False

    def _save_columns(
        self, table_name: str, columns: list[dict[str, Any]], db_name: str = "", update_existing: bool = False
    ) -> bool:
        """
        Save columns information to common_columns.csv and columns of tables to table_columns.csv

        Args:
            table_name (str): Table name
            columns (List[Dict[str, Any]]): List of column information
            db_name (str): Database name
            update_existing (bool): Update existing column information

        Returns:
            bool: Whether the save was successful
        """
        full_table_name, db_name, table_name = split_db_table_name(table_name, db_name)
        # Load existing data
        tables_data = self._load_csv_file(self.table_columns_file)
        common_columns_dict = self._load_common_columns()
        table_spec_columns_dict = self._load_table_spec_columns()

        # Create a set of existing table-column combinations
        existing_table_columns = set()
        for row in tables_data:
            if "db_name" in row and "table_name" in row and "column_name" in row:
                key = f"{row['db_name']}.{row['table_name']}:{row['column_name']}"
                existing_table_columns.add(key)

        # Update table_columns.csv and track new columns to add

        for column in columns:
            if "column_name" not in column:
                continue

            column_name = column["column_name"]
            is_common_column = column.get("is_common", False)

            key = f"{full_table_name}:{column_name}"
            column_info = {k: str(v) for k, v in column.items() if k != "is_common"}
            if not is_common_column:
                column_info["db_name"] = db_name
                column_info["table_name"] = table_name

            # New column of the table -> add to table_columns.csv
            if key not in existing_table_columns:
                tables_data.append({"db_name": db_name, "table_name": table_name, "column_name": column_name})
                existing_table_columns.add(key)
                if is_common_column:
                    # Handle common_columns.csv - avoid duplicates
                    if column_name not in common_columns_dict:
                        # Add new columns to columns_data
                        logger.info(f"Add new column column {column_name}")
                        common_columns_dict[column_name] = column_info
                else:
                    table_spec_columns_dict[key] = column_info
            # Apply updates to existing columns in columns_data
            elif update_existing:
                if is_common_column:
                    common_columns_dict[column_name] = column_info
                else:
                    table_spec_columns_dict[key] = column_info

        # Save updated data
        tables_success = self._save_csv_file(
            self.table_columns_file, tables_data, ["db_name", "table_name", "column_name"]
        )
        common_columns_success = self._save_csv_file(
            self.common_columns_file,
            list(common_columns_dict.values()),
            ["column_name", "display_name", "alias", "type", "category", "tag", "description"],
        )
        table_spec_columns_success = self._save_csv_file(
            self.table_spec_columns_file,
            list(table_spec_columns_dict.values()),
            ["db_name", "table_name", "column_name", "display_name", "alias", "type", "category", "tag", "description"],
        )

        success = tables_success and common_columns_success and table_spec_columns_success
        if success:
            # Clear cache to ensure consistency
            self._clear_cache()
            logger.debug(f"Successfully saved columns for table {table_name}")

        return success

    def save_table_sql_examples(self, table: str, examples: list[dict[str, str]], database: str | None = None) -> bool:
        # Validate input data (let validation errors propagate)
        self._validate_table_name(table)
        self._validate_sql_examples(examples)

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)

            sql_examples = self._load_yaml_file(self.sql_example_file)

            # Ensure database exists in structure
            if db_name not in sql_examples:
                sql_examples[db_name] = {}

            # example text
            example_text = ""
            for example in examples:
                example_text += f"Q: {example['question']}\nA: {example['answer']}\n\n"

            sql_examples[db_name][table_name] = example_text.strip()

            success = self._save_yaml_file(self.sql_example_file, sql_examples)

            if success:
                logger.info(f"Successfully saved {len(examples)} examples for table {full_table_name}")
                # Update cache
                self._sql_example_cache = sql_examples

            return success
        except Exception as e:
            logger.error(f"Unexpected error when saving table examples: {e}")
            logger.error(traceback.format_stack())
            return False

    def save_table_selection_examples(self, examples: list[tuple[str, list[str]]]) -> bool:
        example_data = []
        for example in examples:
            example_data.append({"question": example[0], "selected_tables": example[1]})
        save_success = self._save_csv_file(
            self.table_selection_example_file, example_data, ["question", "selected_tables"]
        )
        if save_success:
            logger.info(f"Successfully saved {len(examples)} table selection examples.")
        return save_success

    def check_exists(self) -> bool:
        try:
            # Check if essential catalog files exist and have content
            files_missing = (
                not os.path.exists(self.table_columns_file)
                or not os.path.exists(self.common_columns_file)
                or os.path.getsize(self.table_columns_file) <= 1  # Empty or just header
                or os.path.getsize(self.common_columns_file) <= 1
            )

            return not files_missing

        except Exception as e:
            logger.warning(f"Error checking catalog existence: {e}")
            logger.error(traceback.format_stack())
            return False
