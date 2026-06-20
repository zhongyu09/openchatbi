"""Tests for knowledge and schema discovery tools."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openchatbi.tool import search_knowledge as sk
from openchatbi.tool.search_knowledge import search_knowledge, search_schema, show_schema


class TestSearchKnowledge:
    """Knowledge search is field/business/example oriented, not table discovery."""

    def test_search_knowledge_columns_returns_field_metadata(self):
        with patch.object(sk, "get_relevant_columns", return_value=["user_id", "revenue"]):
            with patch.dict(
                sk.col_dict,
                {
                    "user_id": {
                        "column_name": "user_id",
                        "display_name": "User ID",
                        "category": "dimension",
                        "type": "bigint",
                        "description": "Unique user identifier",
                        "alias": ["uid"],
                    },
                    "revenue": {
                        "column_name": "revenue",
                        "display_name": "Revenue",
                        "category": "metric",
                        "type": "decimal",
                        "description": "Revenue amount",
                    },
                },
                clear=True,
            ):
                result = search_knowledge.invoke(
                    {
                        "reasoning": "explain fields",
                        "query_list": ["user", "revenue"],
                        "knowledge_bases": ["columns"],
                    }
                )

        assert result["columns"] == [
            {
                "column_name": "revenue",
                "display_name": "Revenue",
                "category": "metric",
                "type": "decimal",
                "description": "Revenue amount",
                "alias": "",
            },
            {
                "column_name": "user_id",
                "display_name": "User ID",
                "category": "dimension",
                "type": "bigint",
                "description": "Unique user identifier",
                "alias": ["uid"],
            },
        ]
        assert result["warnings"] == []

    def test_search_knowledge_does_not_expose_related_tables(self):
        with patch.object(sk, "get_relevant_columns", return_value=["order_id"]):
            with (
                patch.dict(
                    sk.col_dict,
                    {
                        "order_id": {
                            "column_name": "order_id",
                            "display_name": "Order ID",
                            "category": "dimension",
                            "type": "bigint",
                            "description": "Order identifier",
                        }
                    },
                    clear=True,
                ),
                patch.dict(sk.column_tables_mapping, {"order_id": ["mart.orders"]}, clear=True),
            ):
                result = search_knowledge.invoke(
                    {
                        "reasoning": "field meaning only",
                        "query_list": ["order"],
                        "knowledge_bases": ["columns"],
                    }
                )

        assert "mart.orders" not in str(result["columns"])
        assert result["warnings"] == []

    def test_search_knowledge_sql_examples(self):
        fake_store = Mock()
        fake_store.retrieve.return_value = [("how many users", "SELECT COUNT(*) FROM users", ["users"])]

        with patch.object(sk, "get_learned_sql_store", return_value=fake_store):
            result = search_knowledge.invoke(
                {
                    "reasoning": "need examples",
                    "query_list": ["user count"],
                    "knowledge_bases": ["sql_examples"],
                }
            )

        assert "sql_examples" in result
        assert "SELECT COUNT(*) FROM users" in result["sql_examples"]
        assert result["warnings"] == []
        fake_store.retrieve.assert_called_once()

    def test_search_knowledge_business_without_dedicated_store(self):
        fake_config = SimpleNamespace(bi_config={})

        with patch.object(sk.config, "get", return_value=fake_config):
            result = search_knowledge.invoke(
                {
                    "reasoning": "business term",
                    "query_list": ["revenue"],
                    "knowledge_bases": ["business"],
                }
            )

        assert "business" in result
        assert "no dedicated business knowledge available" in result["business"]
        assert "No dedicated business knowledge available" in result["warnings"][0]

    def test_search_knowledge_rejects_invalid_knowledge_base(self):
        with pytest.raises(Exception):
            search_knowledge.invoke(
                {
                    "reasoning": "invalid kb",
                    "query_list": ["orders"],
                    "knowledge_bases": ["not_supported"],
                }
            )


class TestSearchSchema:
    """Schema discovery is table-centric and structured."""

    def test_search_schema_returns_candidate_tables(self):
        fake_catalog = Mock()
        fake_catalog.get_table_information.return_value = {
            "description": "Orders fact table",
            "selection_rule": "Use for order analysis",
        }
        fake_catalog.get_column_list.return_value = [
            {
                "column_name": "order_id",
                "display_name": "Order ID",
                "category": "dimension",
                "type": "bigint",
                "description": "Order identifier",
            },
            {
                "column_name": "order_date",
                "display_name": "Order Date",
                "category": "dimension",
                "type": "date",
                "description": "Date the order was placed",
            },
            {
                "column_name": "order_total",
                "display_name": "Order Total",
                "category": "metric",
                "type": "decimal",
                "description": "Order revenue",
            },
        ]

        with (
            patch.object(sk.config, "get", return_value=SimpleNamespace(catalog_store=fake_catalog)),
            patch.object(sk, "get_relevant_columns", return_value=["order_id", "order_date"]),
            patch.dict(
                sk.column_tables_mapping,
                {"order_id": ["mart.orders"], "order_date": ["mart.orders"]},
                clear=True,
            ),
        ):
            result = search_schema.invoke(
                {
                    "reasoning": "find order schema",
                    "query_list": ["orders", "daily order count"],
                    "metrics": ["order count"],
                }
            )

        assert result["matched_column_count"] == 2
        assert result["candidates"][0]["table"] == "mart.orders"
        assert result["candidates"][0]["matched_columns"] == ["order_date", "order_id"]
        assert result["candidates"][0]["date_columns"] == ["order_date"]
        assert result["candidates"][0]["metric_columns"] == ["order_total"]
        assert result["candidates"][0]["dimension_columns"] == ["order_id", "order_date"]
        assert result["candidates"][0]["columns"][0]["column_name"] == "order_id"
        assert result["warnings"] == []

    def test_search_schema_respects_max_tables(self):
        fake_catalog = Mock()
        fake_catalog.get_table_information.side_effect = lambda table: {"description": table}
        fake_catalog.get_column_list.return_value = [
            {
                "column_name": "order_id",
                "display_name": "Order ID",
                "category": "dimension",
                "type": "bigint",
                "description": "Order identifier",
            }
        ]

        with (
            patch.object(sk.config, "get", return_value=SimpleNamespace(catalog_store=fake_catalog)),
            patch.object(sk, "get_relevant_columns", return_value=["order_id"]),
            patch.dict(sk.column_tables_mapping, {"order_id": ["a.orders", "b.orders"]}, clear=True),
        ):
            result = search_schema.invoke(
                {
                    "reasoning": "limit candidates",
                    "query_list": ["orders"],
                    "max_tables": 1,
                }
            )

        assert len(result["candidates"]) == 1
        assert "Returned top 1 tables out of 2 candidates." in result["warnings"]

    def test_search_schema_no_matches(self):
        fake_catalog = Mock()

        with (
            patch.object(sk.config, "get", return_value=SimpleNamespace(catalog_store=fake_catalog)),
            patch.object(sk, "get_relevant_columns", return_value=[]),
            patch.dict(sk.column_tables_mapping, {}, clear=True),
        ):
            result = search_schema.invoke(
                {
                    "reasoning": "unknown schema",
                    "query_list": ["not-a-real-term"],
                }
            )

        assert result["candidates"] == []
        assert result["unmatched_terms"] == ["not-a-real-term"]
        assert "No relevant columns matched the schema query terms." in result["warnings"]
        assert "No schema candidates found for the provided query terms." in result["warnings"]


class TestShowSchema:
    """Known table schema inspection returns full structured metadata."""

    def test_show_schema_returns_structured_table_details(self):
        fake_catalog = Mock()
        fake_catalog.get_table_information.return_value = {
            "description": "Orders fact table",
            "selection_rule": "Use for order analysis",
            "sql_rule": "Filter by order_date",
            "derived_metric": "order_count = COUNT(order_id)",
        }
        fake_catalog.get_column_list.return_value = [
            {
                "column_name": "order_id",
                "display_name": "Order ID",
                "category": "dimension",
                "type": "bigint",
                "description": "Order identifier",
            },
            {
                "column_name": "order_date",
                "display_name": "Order Date",
                "category": "dimension",
                "type": "date",
                "description": "Order date",
            },
        ]

        with patch.object(sk.config, "get", return_value=SimpleNamespace(catalog_store=fake_catalog)):
            result = show_schema.invoke({"reasoning": "inspect", "tables": ["mart.orders"]})

        schema = result["schemas"][0]
        assert schema["table"] == "mart.orders"
        assert schema["selection_rule"] == "Use for order analysis"
        assert schema["sql_rule"] == "Filter by order_date"
        assert schema["derived_metric"] == "order_count = COUNT(order_id)"
        assert schema["date_columns"] == ["order_date"]
        assert schema["dimension_columns"] == ["order_id", "order_date"]
        assert result["missing_tables"] == []
        assert result["warnings"] == []

    def test_show_schema_reports_missing_tables(self):
        fake_catalog = Mock()
        fake_catalog.get_table_information.return_value = {}

        with patch.object(sk.config, "get", return_value=SimpleNamespace(catalog_store=fake_catalog)):
            result = show_schema.invoke({"reasoning": "inspect", "tables": ["missing.table"]})

        assert result["schemas"] == []
        assert result["missing_tables"] == ["missing.table"]
        assert "Some tables were not found in catalog" in result["warnings"][0]
