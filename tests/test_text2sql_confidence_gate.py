"""Tests for HITL confidence scoring node and confidence gate."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from openchatbi.config_loader import Config
from openchatbi.constants import SQL_SUCCESS
from openchatbi.graph_state import SQLGraphState, SQLOutputState
from openchatbi.text2sql.confidence import ConfidenceResult


class TestConfidenceGateConfig:
    def test_confidence_flags_default_off(self):
        cfg = Config(default_llm=MagicMock(), data_warehouse_config={"uri": "sqlite:///:memory:"})
        assert cfg.enable_confidence_gate is False
        assert cfg.sql_confidence_threshold == 0.7
        assert cfg.confidence_gate_mode == "post_exec"

    def test_confidence_flags_from_dict(self):
        cfg = Config.from_dict(
            {
                "default_llm": MagicMock(),
                "data_warehouse_config": {"uri": "sqlite:///:memory:"},
                "enable_confidence_gate": True,
                "sql_confidence_threshold": 0.5,
                "confidence_gate_mode": "pre_exec",
            }
        )
        assert cfg.enable_confidence_gate is True
        assert cfg.sql_confidence_threshold == 0.5
        assert cfg.confidence_gate_mode == "pre_exec"


class TestConfidenceStateFields:
    def test_sqlgraphstate_accepts_confidence_fields(self):
        state = SQLGraphState(
            messages=[HumanMessage(content="q")],
            sql="SELECT 1",
            sql_confidence=0.42,
            confidence_reasons=["WHERE clause missing filter"],
            human_sql_decision="approve",
        )
        assert state["sql_confidence"] == 0.42
        assert state["confidence_reasons"] == ["WHERE clause missing filter"]
        assert state["human_sql_decision"] == "approve"

    def test_sqloutputstate_exposes_confidence_fields(self):
        # SQLOutputState is the subgraph output schema; fields absent here are
        # filtered out at the subgraph boundary and never reach the parent graph.
        assert "sql_confidence" in SQLOutputState.__annotations__
        assert "confidence_reasons" in SQLOutputState.__annotations__
        assert "human_sql_decision" in SQLOutputState.__annotations__


class TestCreateSqlNodesArity:
    def test_create_sql_nodes_returns_six_callables(self):
        from openchatbi.text2sql.generate_sql import create_sql_nodes

        nodes = create_sql_nodes(Mock(), Mock(), "presto")
        # Convention #2: Task 11 onward, create_sql_nodes is a 6-tuple.
        assert len(nodes) == 6
        assert all(callable(n) for n in nodes)


class TestScoreSqlNode:
    def _nodes(self, mock_llm, mock_catalog):
        from openchatbi.text2sql.generate_sql import create_sql_nodes

        return create_sql_nodes(mock_llm, mock_catalog, "presto")

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="SELECT * FROM users")
        return llm

    @pytest.fixture
    def mock_catalog(self):
        return Mock()

    def test_score_sql_node_returns_confidence(self, mock_llm, mock_catalog):
        nodes = self._nodes(mock_llm, mock_catalog)
        # create_sql_nodes now returns 6 callables (was 4): + score_sql, confidence_gate
        score_sql_node = nodes[4]
        fake_result = ConfidenceResult(
            score=0.35, reasons=["WHERE missing"], checks={"where": False}
        )
        with patch(
            "openchatbi.text2sql.generate_sql.SimpleSQLEvaluator"
        ) as MockEval:
            MockEval.return_value.evaluate.return_value = fake_result
            state = SQLGraphState(
                messages=[],
                rewrite_question="how many users",
                sql="SELECT * FROM users",
                schema_info={"columns": ["id"]},
                data="id\n1\n",
                sql_execution_result=SQL_SUCCESS,
            )
            out = score_sql_node(state)
        assert out["sql_confidence"] == 0.35
        assert out["confidence_reasons"] == ["WHERE missing"]

    def test_score_sql_node_skips_on_failed_sql(self, mock_llm, mock_catalog):
        nodes = self._nodes(mock_llm, mock_catalog)
        score_sql_node = nodes[4]
        state = SQLGraphState(
            messages=[], rewrite_question="q", sql="SELECT 1", sql_execution_result="SQL_SYNTAX_ERROR"
        )
        out = score_sql_node(state)
        # No confidence computed for non-success executions.
        assert out == {}


class TestRouteAfterConfidence:
    def test_route_approve_goes_to_visualization(self):
        from openchatbi.text2sql.sql_graph import route_after_confidence

        assert route_after_confidence({"human_sql_decision": "approve"}) == "generate_visualization"

    def test_route_reject_goes_to_regenerate(self):
        from openchatbi.text2sql.sql_graph import route_after_confidence

        assert route_after_confidence({"human_sql_decision": "reject"}) == "regenerate_sql"

    def test_route_edit_goes_to_execute(self):
        from openchatbi.text2sql.sql_graph import route_after_confidence

        # An edited SQL must be re-executed before visualization.
        assert route_after_confidence({"human_sql_decision": "edit"}) == "execute_sql"

    def test_route_default_when_no_decision(self):
        from openchatbi.text2sql.sql_graph import route_after_confidence

        assert route_after_confidence({}) == "generate_visualization"
