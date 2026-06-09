"""Tests for Golden-SQL capture flow and sql_examples KB branch."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from openchatbi.config_loader import Config


class TestGoldenSqlConfig:
    def test_golden_flags_default_off(self):
        cfg = Config(default_llm=MagicMock(), data_warehouse_config={"uri": "sqlite:///:memory:"})
        assert cfg.enable_golden_sql is False
        assert cfg.golden_sql_namespace == "global"

    def test_golden_flags_from_dict(self):
        cfg = Config.from_dict(
            {
                "default_llm": MagicMock(),
                "data_warehouse_config": {"uri": "sqlite:///:memory:"},
                "enable_golden_sql": True,
                "golden_sql_namespace": "team_a",
            }
        )
        assert cfg.enable_golden_sql is True
        assert cfg.golden_sql_namespace == "team_a"


class TestAppendSqlExample:
    def test_append_adds_new_example_without_overwriting(self, mock_catalog_store):
        # save_table_sql_examples overwrites; append_sql_example must keep prior ones.
        mock_catalog_store.save_table_sql_examples(
            "test.test_table", [{"question": "count rows", "answer": "SELECT COUNT(*) FROM test_table"}]
        )
        ok = mock_catalog_store.append_sql_example(
            "how many names", "SELECT COUNT(name) FROM test_table", ["test.test_table"], source="golden"
        )
        assert ok is True
        examples = mock_catalog_store.get_sql_examples()
        questions = {q for q, _sql, _t in examples}
        assert "count rows" in questions
        assert "how many names" in questions

    def test_append_dedups_identical_question(self, mock_catalog_store):
        mock_catalog_store.append_sql_example(
            "dup q", "SELECT 1 FROM test_table", ["test.test_table"], source="golden"
        )
        mock_catalog_store.append_sql_example(
            "dup q", "SELECT 1 FROM test_table", ["test.test_table"], source="golden"
        )
        examples = mock_catalog_store.get_sql_examples()
        dup = [q for q, _sql, _t in examples if q == "dup q"]
        assert len(dup) == 1


class TestGoldenCaptureOnApprove:
    def _gate_node(self, mock_catalog_store):
        from openchatbi.text2sql.generate_sql import create_sql_nodes

        llm = Mock()
        nodes = create_sql_nodes(llm, mock_catalog_store, "presto")
        return nodes[5]  # confidence_gate_node

    def test_approve_dual_writes_golden_sql(self, mock_catalog_store):
        from openchatbi.config_loader import Config
        from openchatbi.constants import SQL_SUCCESS
        from openchatbi.graph_state import SQLGraphState

        cfg = Config(
            default_llm=MagicMock(),
            data_warehouse_config={"uri": "sqlite:///:memory:"},
            catalog_store=mock_catalog_store,
            enable_confidence_gate=True,
            sql_confidence_threshold=0.7,
            enable_golden_sql=True,
            golden_sql_namespace="global",
        )
        learned_store = Mock()
        gate = self._gate_node(mock_catalog_store)
        state = SQLGraphState(
            messages=[],
            rewrite_question="how many names",
            sql="SELECT COUNT(name) FROM test_table",
            tables=[{"table": "test.test_table"}],
            sql_confidence=0.95,  # >= threshold -> auto-approve, no interrupt
            sql_execution_result=SQL_SUCCESS,
        )
        with patch("openchatbi.text2sql.generate_sql.config.get", return_value=cfg), patch(
            "openchatbi.text2sql.generate_sql.get_learned_sql_store", return_value=learned_store
        ):
            out = gate(state)
        assert out["human_sql_decision"] == "approve"
        # Vector-store write (S3) ...
        learned_store.add_golden_sql.assert_called_once()
        # ... and durable YAML write (catalog) both happened.
        examples = mock_catalog_store.get_sql_examples()
        assert any(q == "how many names" for q, _sql, _t in examples)

    def test_approve_skips_capture_when_golden_disabled(self, mock_catalog_store):
        from openchatbi.config_loader import Config
        from openchatbi.constants import SQL_SUCCESS
        from openchatbi.graph_state import SQLGraphState

        cfg = Config(
            default_llm=MagicMock(),
            data_warehouse_config={"uri": "sqlite:///:memory:"},
            catalog_store=mock_catalog_store,
            enable_confidence_gate=True,
            sql_confidence_threshold=0.7,
            enable_golden_sql=False,
        )
        learned_store = Mock()
        gate = self._gate_node(mock_catalog_store)
        state = SQLGraphState(
            messages=[],
            rewrite_question="q",
            sql="SELECT 1 FROM test_table",
            tables=[{"table": "test.test_table"}],
            sql_confidence=0.95,
            sql_execution_result=SQL_SUCCESS,
        )
        with patch("openchatbi.text2sql.generate_sql.config.get", return_value=cfg), patch(
            "openchatbi.text2sql.generate_sql.get_learned_sql_store", return_value=learned_store
        ):
            gate(state)
        learned_store.add_golden_sql.assert_not_called()


class TestSearchKnowledgeSqlExamples:
    def test_sql_examples_branch_returns_retrieved_examples(self):
        from openchatbi.tool import search_knowledge as sk

        fake_store = Mock()
        fake_store.retrieve.return_value = [
            ("how many users", "SELECT COUNT(*) FROM users", ["users"]),
        ]
        with patch.object(sk, "get_learned_sql_store", return_value=fake_store):
            result = sk.search_knowledge.invoke(
                {
                    "reasoning": "need examples",
                    "query_list": ["user count"],
                    "knowledge_bases": ["sql_examples"],
                    "with_table_list": False,
                }
            )
        assert "sql_examples" in result
        assert "SELECT COUNT(*) FROM users" in result["sql_examples"]
        fake_store.retrieve.assert_called()

    def test_business_branch_now_implemented(self):
        # Previously 'business' was documented but never branched; it must now
        # at least return a (possibly empty) keyed entry, not silently drop.
        from openchatbi.tool import search_knowledge as sk

        with patch.object(sk, "get_learned_sql_store", return_value=None), patch.object(
            sk, "_search_column_from_catalog", return_value="revenue: Revenue amount"
        ):
            result = sk.search_knowledge.invoke(
                {
                    "reasoning": "biz",
                    "query_list": ["revenue"],
                    "knowledge_bases": ["business"],
                    "with_table_list": False,
                }
            )
        assert "business" in result

    def test_sql_examples_branch_none_store_no_op(self):
        """When learned SQL store is None, sql_examples returns a graceful message."""
        from openchatbi.tool import search_knowledge as sk

        with patch.object(sk, "get_learned_sql_store", return_value=None):
            result = sk.search_knowledge.invoke(
                {
                    "reasoning": "need examples",
                    "query_list": ["count"],
                    "knowledge_bases": ["sql_examples"],
                    "with_table_list": False,
                }
            )
        assert "sql_examples" in result
        assert "no learned SQL store available" in result["sql_examples"]
