"""Tests for text2sql schema linking functionality."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from openchatbi.graph_state import SQLGraphState
from openchatbi.text2sql.schema_linking import schema_linking


class TestText2SQLSchemaLinking:
    """Test text2sql schema linking functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = Mock()
        llm.invoke.return_value = AIMessage(content='{"tables": [{"table": "users", "reason": "Contains user data"}]}')
        return llm

    @pytest.fixture
    def mock_catalog(self):
        """Mock catalog store for testing."""
        catalog = Mock()
        catalog.get_table_information.return_value = {
            "description": "User data table",
            "selection_rule": "Use for user-related queries",
        }
        return catalog

    def test_select_table_function_creation(self, mock_llm, mock_catalog):
        """Test creating table selection function."""
        select_func = schema_linking(mock_llm, mock_catalog)

        assert callable(select_func)

    def test_select_table_success(self, mock_llm, mock_catalog):
        """Test successful table selection."""
        with patch("openchatbi.text2sql.schema_linking.get_relevant_columns") as mock_get_columns:
            mock_get_columns.return_value = ["user_id", "name", "email"]

            with patch(
                "openchatbi.text2sql.schema_linking.column_tables_mapping",
                {"user_id": ["users", "profiles"], "name": ["users"], "email": ["users", "contacts"]},
            ):
                with patch(
                    "openchatbi.text2sql.schema_linking.col_dict",
                    {
                        "user_id": {
                            "column_name": "user_id",
                            "category": "dimension",
                            "display_name": "User ID",
                            "description": "Unique user identifier",
                        },
                        "name": {
                            "column_name": "name",
                            "category": "dimension",
                            "display_name": "Name",
                            "description": "User full name",
                        },
                        "email": {
                            "column_name": "email",
                            "category": "dimension",
                            "display_name": "Email",
                            "description": "User email address",
                        },
                    },
                ):
                    with patch("openchatbi.text2sql.schema_linking.table_selection_retriever") as mock_retriever:
                        mock_retriever.invoke.return_value = []

                        with patch("openchatbi.text2sql.schema_linking.table_selection_example_dict", {}):
                            with patch("openchatbi.text2sql.schema_linking.extract_json_from_answer") as mock_extract:
                                mock_extract.return_value = {
                                    "tables": [{"table": "users", "reason": "Contains user data"}]
                                }

                                select_func = schema_linking(mock_llm, mock_catalog)

                                state = SQLGraphState(
                                    messages=[],
                                    question="Show user information",
                                    rewrite_question="Show user information",
                                    info_entities={
                                        "keywords": ["user", "information"],
                                        "dimensions": ["name", "email"],
                                        "metrics": [],
                                    },
                                )

                                result = select_func(state)

        assert "tables" in result
        assert len(result["tables"]) == 1
        assert result["tables"][0]["table"] == "users"

    def test_select_table_missing_rewrite_question(self, mock_llm, mock_catalog):
        """Test table selection with missing rewrite question."""
        select_func = schema_linking(mock_llm, mock_catalog)

        state = SQLGraphState(
            messages=[],
            question="Show user information",
            # Missing rewrite_question
        )

        result = select_func(state)
        assert result == {}

    def test_select_table_with_examples(self, mock_llm, mock_catalog):
        """Test table selection with similar examples."""
        with patch("openchatbi.text2sql.schema_linking.get_relevant_columns") as mock_get_columns:
            mock_get_columns.return_value = ["user_id", "revenue"]

            with patch(
                "openchatbi.text2sql.schema_linking.column_tables_mapping", {"user_id": ["users"], "revenue": ["sales"]}
            ):
                with patch(
                    "openchatbi.text2sql.schema_linking.col_dict",
                    {
                        "user_id": {
                            "column_name": "user_id",
                            "category": "dimension",
                            "display_name": "User ID",
                            "description": "Unique user identifier",
                        },
                        "revenue": {
                            "column_name": "revenue",
                            "category": "metric",
                            "display_name": "Revenue",
                            "description": "Total revenue amount",
                        },
                    },
                ):
                    # Mock similar examples
                    mock_document = Mock()
                    mock_document.page_content = "What is user revenue?"

                    with patch("openchatbi.text2sql.schema_linking.table_selection_retriever") as mock_retriever:
                        mock_retriever.invoke.return_value = [mock_document]

                        with patch(
                            "openchatbi.text2sql.schema_linking.table_selection_example_dict",
                            {"What is user revenue?": ["users", "sales"]},
                        ):
                            with patch("openchatbi.text2sql.schema_linking.extract_json_from_answer") as mock_extract:
                                mock_extract.return_value = {"tables": [{"table": "users"}, {"table": "sales"}]}

                                select_func = schema_linking(mock_llm, mock_catalog)

                                state = SQLGraphState(
                                    messages=[],
                                    question="Show user revenue",
                                    rewrite_question="Show user revenue",
                                    info_entities={
                                        "keywords": ["user", "revenue"],
                                        "dimensions": ["user_id"],
                                        "metrics": ["revenue"],
                                    },
                                )

                                result = select_func(state)

        assert "tables" in result
        assert len(result["tables"]) == 2

    def test_select_table_invalid_table_selection(self, mock_llm, mock_catalog):
        """Test handling of invalid table selection."""
        with patch("openchatbi.text2sql.schema_linking.get_relevant_columns") as mock_get_columns:
            mock_get_columns.return_value = ["user_id"]

            with patch("openchatbi.text2sql.schema_linking.column_tables_mapping", {"user_id": ["users"]}):
                with patch(
                    "openchatbi.text2sql.schema_linking.col_dict",
                    {
                        "user_id": {
                            "column_name": "user_id",
                            "category": "dimension",
                            "display_name": "User ID",
                            "description": "Unique user identifier",
                        }
                    },
                ):
                    with patch("openchatbi.text2sql.schema_linking.table_selection_retriever") as mock_retriever:
                        mock_retriever.invoke.return_value = []

                        with patch("openchatbi.text2sql.schema_linking.table_selection_example_dict", {}):
                            with patch("openchatbi.text2sql.schema_linking.extract_json_from_answer") as mock_extract:
                                # Return invalid table not in candidate list
                                mock_extract.return_value = {"tables": [{"table": "invalid_table"}]}

                                select_func = schema_linking(mock_llm, mock_catalog)

                                state = SQLGraphState(
                                    messages=[],
                                    question="Show user info",
                                    rewrite_question="Show user info",
                                    info_entities={"keywords": ["user"], "dimensions": ["user_id"], "metrics": []},
                                )

                                result = select_func(state)

        # Should return empty dict when invalid table selected
        assert result == {}

    def test_select_table_retry_mechanism(self, mock_llm, mock_catalog):
        """Test retry mechanism for table selection."""
        with patch("openchatbi.text2sql.schema_linking.get_relevant_columns") as mock_get_columns:
            mock_get_columns.return_value = ["user_id"]

            with patch("openchatbi.text2sql.schema_linking.column_tables_mapping", {"user_id": ["users"]}):
                with patch(
                    "openchatbi.text2sql.schema_linking.col_dict",
                    {
                        "user_id": {
                            "column_name": "user_id",
                            "category": "dimension",
                            "display_name": "User ID",
                            "description": "Unique user identifier",
                        }
                    },
                ):
                    with patch("openchatbi.text2sql.schema_linking.table_selection_retriever") as mock_retriever:
                        mock_retriever.invoke.return_value = []

                        with patch("openchatbi.text2sql.schema_linking.table_selection_example_dict", {}):
                            with patch("openchatbi.text2sql.schema_linking.extract_json_from_answer") as mock_extract:
                                # First returns invalid, then valid
                                mock_extract.side_effect = [
                                    {"tables": [{"table": "invalid_table"}]},
                                    {"tables": [{"table": "users"}]},
                                ]

                                select_func = schema_linking(mock_llm, mock_catalog)

                                state = SQLGraphState(
                                    messages=[],
                                    question="Show user info",
                                    rewrite_question="Show user info",
                                    info_entities={"keywords": ["user"], "dimensions": ["user_id"], "metrics": []},
                                )

                                result = select_func(state)

        assert "tables" in result
        assert result["tables"][0]["table"] == "users"

    def test_select_table_with_time_filter(self, mock_llm, mock_catalog):
        """Test table selection with time filtering."""
        # Mock table with start_time
        mock_catalog.get_table_information.return_value = {
            "description": "User data table",
            "selection_rule": "Use for user queries",
            "start_time": "2024-01-01",
        }

        with patch("openchatbi.text2sql.schema_linking.get_relevant_columns") as mock_get_columns:
            mock_get_columns.return_value = ["user_id"]

            with patch("openchatbi.text2sql.schema_linking.column_tables_mapping", {"user_id": ["users"]}):
                with patch(
                    "openchatbi.text2sql.schema_linking.col_dict",
                    {
                        "user_id": {
                            "column_name": "user_id",
                            "category": "dimension",
                            "display_name": "User ID",
                            "description": "Unique user identifier",
                        }
                    },
                ):
                    with patch("openchatbi.text2sql.schema_linking.table_selection_retriever") as mock_retriever:
                        mock_retriever.invoke.return_value = []

                        with patch("openchatbi.text2sql.schema_linking.table_selection_example_dict", {}):
                            with patch("openchatbi.text2sql.schema_linking.extract_json_from_answer") as mock_extract:
                                mock_extract.return_value = {"tables": [{"table": "users"}]}

                                select_func = schema_linking(mock_llm, mock_catalog)

                                state = SQLGraphState(
                                    messages=[],
                                    question="Show recent user info",
                                    rewrite_question="Show recent user info",
                                    info_entities={
                                        "keywords": ["user"],
                                        "dimensions": ["user_id"],
                                        "metrics": [],
                                        "start_time": "2024-06-01",  # Later than table start_time
                                    },
                                )

                                result = select_func(state)

        assert "tables" in result
        assert result["tables"][0]["table"] == "users"

    def test_select_table_llm_error_handling(self, mock_llm, mock_catalog):
        """Test handling of LLM errors during table selection."""
        mock_llm.invoke.side_effect = Exception("LLM service error")

        with patch("openchatbi.text2sql.schema_linking.get_relevant_columns") as mock_get_columns:
            mock_get_columns.return_value = ["user_id"]

            with patch("openchatbi.text2sql.schema_linking.column_tables_mapping", {"user_id": ["users"]}):
                with patch(
                    "openchatbi.text2sql.schema_linking.col_dict",
                    {
                        "user_id": {
                            "column_name": "user_id",
                            "category": "dimension",
                            "display_name": "User ID",
                            "description": "Unique user identifier",
                        }
                    },
                ):
                    with patch("openchatbi.text2sql.schema_linking.table_selection_retriever") as mock_retriever:
                        mock_retriever.invoke.return_value = []

                        select_func = schema_linking(mock_llm, mock_catalog)

                        state = SQLGraphState(
                            messages=[],
                            question="Show user info",
                            rewrite_question="Show user info",
                            info_entities={"keywords": ["user"], "dimensions": ["user_id"], "metrics": []},
                        )

                        result = select_func(state)

        # Should handle error gracefully and return empty dict
        assert result == {}

    def test_select_table_max_retries_exceeded(self, mock_llm, mock_catalog):
        """Test behavior when max retries are exceeded."""
        with patch("openchatbi.text2sql.schema_linking.get_relevant_columns") as mock_get_columns:
            mock_get_columns.return_value = ["user_id"]

            with patch("openchatbi.text2sql.schema_linking.column_tables_mapping", {"user_id": ["users"]}):
                with patch(
                    "openchatbi.text2sql.schema_linking.col_dict",
                    {
                        "user_id": {
                            "column_name": "user_id",
                            "category": "dimension",
                            "display_name": "User ID",
                            "description": "Unique user identifier",
                        }
                    },
                ):
                    with patch("openchatbi.text2sql.schema_linking.table_selection_retriever") as mock_retriever:
                        mock_retriever.invoke.return_value = []

                        with patch("openchatbi.text2sql.schema_linking.table_selection_example_dict", {}):
                            with patch("openchatbi.text2sql.schema_linking.extract_json_from_answer") as mock_extract:
                                # Always return invalid table
                                mock_extract.return_value = {"tables": [{"table": "invalid_table"}]}

                                select_func = schema_linking(mock_llm, mock_catalog)

                                state = SQLGraphState(
                                    messages=[],
                                    question="Show user info",
                                    rewrite_question="Show user info",
                                    info_entities={"keywords": ["user"], "dimensions": ["user_id"], "metrics": []},
                                )

                                result = select_func(state)

        # Should return empty dict after max retries
        assert result == {}
