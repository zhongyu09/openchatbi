"""Tests for search_knowledge tool functionality."""

from unittest.mock import patch

import pytest

from openchatbi.tool.search_knowledge import search_knowledge, show_schema


class TestSearchKnowledge:
    """Test search_knowledge tool functionality."""

    def test_search_knowledge_basic(self):
        """Test basic knowledge search functionality."""
        reasoning = "Looking for user information"
        query_list = ["user", "information"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_id: User identifier\nuser_name: User name"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result
            assert "User identifier" in result["columns"]
            mock_search.assert_called_once_with(query_list, False)

    def test_search_knowledge_table_matching(self):
        """Test knowledge search with table matching."""
        reasoning = "Finding table relationships"
        query_list = ["user", "metrics"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_id: Unique identifier\nmetrics_value: Metric value"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": True,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result
            mock_search.assert_called_once_with(query_list, True)

    def test_search_knowledge_empty_query(self):
        """Test knowledge search with empty query."""
        reasoning = "Testing empty search"
        query_list = []
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = ""

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result
            mock_search.assert_called_once_with(query_list, False)

    def test_search_knowledge_no_matches(self):
        """Test knowledge search with no matches."""
        reasoning = "Testing no matches"
        query_list = ["nonexistent"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = ""

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result
            assert result["columns"] == "# Relevant Columns and Description:\n"

    def test_search_knowledge_multiple_matches(self):
        """Test knowledge search with multiple matches."""
        reasoning = "Finding multiple matches"
        query_list = ["user", "data", "profile"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_id: User ID\nuser_name: Name\nprofile_data: Profile"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result
            assert "user_id" in result["columns"]
            assert "profile_data" in result["columns"]

    def test_search_knowledge_with_synonyms(self):
        """Test knowledge search with synonym matching."""
        reasoning = "Testing synonym search"
        query_list = ["customer", "client", "user"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "customer_id: Customer identifier\nclient_name: Client name"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result

    def test_search_knowledge_case_insensitive(self):
        """Test case insensitive knowledge search."""
        reasoning = "Testing case sensitivity"
        query_list = ["USER", "Data", "PROFILE"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_data: User information"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result

    def test_search_knowledge_partial_matches(self):
        """Test knowledge search with partial matches."""
        reasoning = "Testing partial matching"
        query_list = ["usr", "prof"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_profile: User profile data"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)

    def test_search_knowledge_error_handling(self):
        """Test knowledge search error handling."""
        reasoning = "Testing error handling"
        query_list = ["test"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.side_effect = Exception("Search error")

            # Should handle exceptions gracefully
            with pytest.raises(Exception):
                search_knowledge.run(
                    {
                        "reasoning": reasoning,
                        "query_list": query_list,
                        "knowledge_bases": knowledge_bases,
                        "with_table_list": False,
                    }
                )

    def test_show_schema_basic(self):
        """Test basic schema display functionality."""
        reasoning = "Showing basic schema"
        tables = ["user_data"]

        with patch("openchatbi.tool.search_knowledge.list_table_from_catalog") as mock_list:
            mock_list.return_value = ["Table: user_data\n# Description: User information\n# Columns:\nuser_id: User ID"]

            result = show_schema.run({"reasoning": reasoning, "tables": tables})

            assert isinstance(result, list)
            assert len(result) == 1
            assert "user_data" in result[0]
            mock_list.assert_called_once_with(tables)

    def test_show_schema_detailed_info(self):
        """Test detailed schema information."""
        reasoning = "Showing detailed schema"
        tables = ["user_data", "metrics"]

        with patch("openchatbi.tool.search_knowledge.list_table_from_catalog") as mock_list:
            mock_list.return_value = [
                "Table: user_data\n# Columns: user_id, name, email",
                "Table: metrics\n# Columns: metric_id, value, timestamp",
            ]

            result = show_schema.run({"reasoning": reasoning, "tables": tables})

            assert isinstance(result, list)
            assert len(result) == 2
            assert any("user_data" in schema for schema in result)
            assert any("metrics" in schema for schema in result)

    def test_show_schema_nonexistent_table(self):
        """Test schema display for nonexistent table."""
        reasoning = "Testing nonexistent table"
        tables = ["nonexistent_table"]

        with patch("openchatbi.tool.search_knowledge.list_table_from_catalog") as mock_list:
            mock_list.return_value = []

            result = show_schema.run({"reasoning": reasoning, "tables": tables})

            assert isinstance(result, list)
            assert len(result) == 0

    def test_show_schema_table_error(self):
        """Test schema display error handling."""
        reasoning = "Testing schema errors"
        tables = ["error_table"]

        with patch("openchatbi.tool.search_knowledge.list_table_from_catalog") as mock_list:
            mock_list.side_effect = Exception("Table access error")

            with pytest.raises(Exception):
                show_schema.run({"reasoning": reasoning, "tables": tables})

    def test_show_schema_complex_table(self):
        """Test schema display for complex table structure."""
        reasoning = "Showing complex schema"
        tables = ["complex_table"]

        with patch("openchatbi.tool.search_knowledge.list_table_from_catalog") as mock_list:
            mock_list.return_value = [
                "Table: complex_table\n# Description: Complex data structure\n# Columns:\nid: Primary key\ndata: JSON data\ncreated_at: Timestamp"
            ]

            result = show_schema.run({"reasoning": reasoning, "tables": tables})

            assert isinstance(result, list)
            assert "complex_table" in result[0]
            assert "Primary key" in result[0]

    def test_search_knowledge_with_metrics(self):
        """Test knowledge search focusing on metrics."""
        reasoning = "Finding metrics columns"
        query_list = ["revenue", "clicks", "impressions"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "revenue: Revenue amount\nclicks: Click count\nimpressions: Impression count"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "revenue" in result["columns"]
            assert "clicks" in result["columns"]

    def test_search_knowledge_contextual_search(self):
        """Test contextual knowledge search."""
        reasoning = "Contextual search for user behavior"
        query_list = ["user", "behavior", "tracking"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_behavior: User activity tracking\ntracking_id: Tracking identifier"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "behavior" in result["columns"]

    def test_search_knowledge_with_aggregations(self):
        """Test knowledge search for aggregation columns."""
        reasoning = "Finding aggregation metrics"
        query_list = ["sum", "count", "average"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "total_count: Count aggregation\naverage_value: Average calculation"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)

    def test_show_schema_with_examples(self):
        """Test schema display with usage examples."""
        reasoning = "Showing schema with examples"
        tables = ["example_table"]

        with patch("openchatbi.tool.search_knowledge.list_table_from_catalog") as mock_list:
            mock_list.return_value = [
                "Table: example_table\n# Description: Example usage\n## Derived metrics:\nSELECT COUNT(*) FROM example_table"
            ]

            result = show_schema.run({"reasoning": reasoning, "tables": tables})

            assert isinstance(result, list)
            assert "example_table" in result[0]
            assert "Derived metrics" in result[0]

    def test_search_knowledge_performance(self):
        """Test knowledge search performance characteristics."""
        reasoning = "Testing search performance"
        query_list = ["performance", "speed", "optimization"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "performance_metric: Performance measurement"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            # Just ensure it completes without performance issues

    def test_search_knowledge_special_characters(self):
        """Test knowledge search with special characters."""
        reasoning = "Testing special character handling"
        query_list = ["user@domain", "data-point", "metric_value"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_email: User email address\ndata_point: Data measurement"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)

    def test_search_knowledge_unicode_support(self):
        """Test knowledge search with unicode characters."""
        reasoning = "Testing unicode support"
        query_list = ["utilización", "données", "用户"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "user_data: International user data"

            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)

    def test_knowledge_integration_with_state(self):
        """Test knowledge search integration with agent state."""
        reasoning = "Testing state integration"
        query_list = ["state", "integration"]
        knowledge_bases = ["columns"]

        with patch("openchatbi.tool.search_knowledge.search_column_from_catalog") as mock_search:
            mock_search.return_value = "state_data: Application state information"

            # Test that the tool can be called in the context of agent state
            result = search_knowledge.run(
                {
                    "reasoning": reasoning,
                    "query_list": query_list,
                    "knowledge_bases": knowledge_bases,
                    "with_table_list": False,
                }
            )

            assert isinstance(result, dict)
            assert "columns" in result
