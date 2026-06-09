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
