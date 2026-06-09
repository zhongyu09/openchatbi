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
