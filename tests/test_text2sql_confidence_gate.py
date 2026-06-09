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


class TestInterruptThroughToolBoundary:
    """Verify the confidence_gate interrupt survives the get_sql_tools
    StructuredTool boundary and that Command(resume=...) routes back to the
    correct node.

    LangGraph 1.x semantics: a dynamic interrupt() raises GraphInterrupt only
    when the interrupting subgraph is invoked from *inside* a parent-graph node
    (the production path: agent's use_tool node -> text2sql StructuredTool ->
    sql_graph.invoke). The subgraph must therefore share the parent's
    checkpointer (compiled without its own). At top level the same interrupt is
    surfaced as ``__interrupt__`` on the returned state instead of raising.
    This test reproduces the nested production path so the GraphInterrupt
    re-raise in agent_graph.call_sql_graph_sync (and the config threading that
    makes resume land on the same thread) are both exercised.
    """

    def _build_gated_subgraph(self):
        from langgraph.graph import START, END, StateGraph
        from openchatbi.graph_state import InputState, SQLGraphState, SQLOutputState
        from openchatbi.text2sql.sql_graph import route_after_confidence
        from openchatbi.text2sql.generate_sql import create_sql_nodes

        llm = Mock()
        llm.invoke.return_value = AIMessage(content="SELECT * FROM users")
        catalog = Mock()
        nodes = create_sql_nodes(llm, catalog, "presto")
        confidence_gate_node = nodes[5]

        def execute_stub(state):
            return {"sql_execution_result": SQL_SUCCESS, "data": "id\n1\n", "schema_info": {}}

        def score_stub(state):
            return {"sql_confidence": 0.30, "confidence_reasons": ["WHERE missing"]}

        def viz_stub(state):
            return {"visualization_dsl": {"chart_type": "bar"}}

        g = StateGraph(SQLGraphState, input_schema=InputState, output_schema=SQLOutputState)
        g.add_node("execute_sql", execute_stub)
        g.add_node("score_sql", score_stub)
        g.add_node("confidence_gate", confidence_gate_node)
        g.add_node("generate_visualization", viz_stub)
        g.add_node("regenerate_sql", lambda s: {"sql": "SELECT 2"})
        g.add_edge(START, "execute_sql")
        g.add_edge("execute_sql", "score_sql")
        g.add_edge("score_sql", "confidence_gate")
        g.add_conditional_edges(
            "confidence_gate",
            route_after_confidence,
            {
                "generate_visualization": "generate_visualization",
                "regenerate_sql": "regenerate_sql",
                "execute_sql": "execute_sql",
            },
        )
        g.add_edge("generate_visualization", END)
        g.add_edge("regenerate_sql", END)
        # No own checkpointer: inherits the parent's so the interrupt propagates
        # as GraphInterrupt into the calling node (production behavior).
        return g.compile()

    def _build_parent_graph(self, tool):
        from langgraph.graph import START, END, StateGraph
        from langgraph.errors import GraphInterrupt
        from langgraph.checkpoint.memory import MemorySaver

        self._tool_raised_interrupt = False

        def call_tool_node(state):
            try:
                result = tool.invoke({"reasoning": "r", "context": "how many users"})
            except GraphInterrupt:
                # text2sql tool re-raises GraphInterrupt (agent_graph.py).
                self._tool_raised_interrupt = True
                raise
            return {"tool_result": result}

        pg = StateGraph(dict)
        pg.add_node("call_tool", call_tool_node)
        pg.add_edge(START, "call_tool")
        pg.add_edge("call_tool", END)
        return pg.compile(checkpointer=MemorySaver())

    def test_low_confidence_interrupts_then_resume_approve(self):
        from openchatbi.agent_graph import get_sql_tools

        subgraph = self._build_gated_subgraph()
        tool = get_sql_tools(subgraph, sync_mode=True)
        parent = self._build_parent_graph(tool)
        cfg = Config(
            default_llm=MagicMock(),
            data_warehouse_config={"uri": "sqlite:///:memory:"},
            enable_confidence_gate=True,
            sql_confidence_threshold=0.7,
        )
        with patch("openchatbi.text2sql.generate_sql.config.get", return_value=cfg):
            run_cfg = {"configurable": {"thread_id": "t-1"}}
            # First run pauses at the confidence gate: the interrupt propagates
            # through the StructuredTool boundary (GraphInterrupt re-raised in
            # call_sql_graph_sync) and surfaces as __interrupt__ on the parent.
            first = parent.invoke({}, config=run_cfg)
            assert self._tool_raised_interrupt is True
            assert "__interrupt__" in first
            interrupt_payload = first["__interrupt__"][0].value
            assert interrupt_payload["buttons"] == ["approve", "reject", "edit"]
            assert "0.30" in interrupt_payload["text"]
            assert parent.get_state(run_cfg).next == ("call_tool",)

            # Resume with the human approval on the same thread -> the gate
            # returns "approve", route_after_confidence -> generate_visualization,
            # and the subgraph (then the tool, then the parent) completes.
            resumed = parent.invoke(Command(resume="approve"), config=run_cfg)
            assert "__interrupt__" not in resumed
            assert "Query Results" in resumed["tool_result"]
            assert parent.get_state(run_cfg).next == ()
