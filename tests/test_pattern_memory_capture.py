"""Tests for gated SQL pattern auto-capture into LearnedSQLStore and blended retrieval.

Convention #2: create_sql_nodes is a 6-tuple (gen, execute, regen, viz, score_sql, gate).
Convention #3: the SQL executor seam is the inner `_execute_sql` closure, driven via
    `mock_catalog.get_sql_engine().connect()` (no invented module-level seam).
Convention #4: the learned store is reached via the `get_learned_sql_store()` accessor.
Convention #5: blended retrieval wraps composite_score in a `(meta, base_rank)` adapter.
Convention #10: auto-capture fires only on terminal success with approval semantics;
    gated on S2 confidence (success != correct) and default-off.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from openchatbi.constants import SQL_SUCCESS
from openchatbi.graph_state import SQLGraphState
from openchatbi.text2sql.confidence import ConfidenceResult
from openchatbi.text2sql.generate_sql import create_sql_nodes


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="SELECT * FROM users")
    return llm


@pytest.fixture
def mock_catalog():
    """Catalog with a working get_sql_engine().connect() context manager returning 1 row."""
    catalog = Mock()
    catalog.get_table_information.return_value = {
        "description": "test table",
        "sql_rule": "",
        "derived_metric": "",
    }
    catalog.get_column_list.return_value = [
        {
            "column_name": "id",
            "type": "bigint",
            "display_name": "Id",
            "description": "pk",
            "alias": "",
        }
    ]
    mock_engine = Mock()
    mock_connection = Mock()
    mock_result = Mock()
    mock_result.fetchall.return_value = [(1,)]
    mock_result.fetchmany.return_value = [(1,)]
    mock_result.keys.return_value = ["c"]
    mock_connection.execute.return_value = mock_result
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_connection
    mock_context_manager.__exit__.return_value = None
    mock_engine.connect.return_value = mock_context_manager
    catalog.get_sql_engine.return_value = mock_engine
    return catalog


def _success_state():
    return SQLGraphState(
        messages=[],
        sql="SELECT 1",
        rewrite_question="how many",
        tables=[{"table": "test_table", "columns": []}],
        sql_retry_count=0,
    )


class TestArity:
    def test_create_sql_nodes_returns_six_callables(self, mock_llm, mock_catalog):
        # Convention #2: arity stays 6 even with the learned store wired.
        nodes = create_sql_nodes(mock_llm, mock_catalog, "presto", learned_sql_store=MagicMock())
        assert len(nodes) == 6
        assert all(callable(n) for n in nodes)


class TestAutoCaptureGate:
    def test_capture_disabled_by_default(self, mock_llm, mock_catalog):
        """No write when enable_pattern_memory is False (default-off)."""
        store = MagicMock()
        _gen, execute_sql_node, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        cfg = MagicMock()
        cfg.enable_pattern_memory = False
        with patch("openchatbi.text2sql.generate_sql.get_memory_config", return_value=cfg):
            out = execute_sql_node(_success_state())

        assert out["sql_execution_result"] == SQL_SUCCESS
        store.add.assert_not_called()

    def test_capture_noop_when_store_is_none(self, mock_llm, mock_catalog):
        """No store wired -> capture is a no-op even with the flag on."""
        _gen, execute_sql_node, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=None
        )
        cfg = MagicMock()
        cfg.enable_pattern_memory = True
        # Should simply not raise; nothing to assert beyond success.
        with patch("openchatbi.text2sql.generate_sql.get_memory_config", return_value=cfg):
            out = execute_sql_node(_success_state())
        assert out["sql_execution_result"] == SQL_SUCCESS

    def test_capture_fires_when_enabled_and_gate_passes(self, mock_llm, mock_catalog):
        """Flag on + confidence >= threshold -> write via store.add(source='auto')."""
        store = MagicMock()
        _gen, execute_sql_node, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.pattern_scope = "global"
        main_cfg = MagicMock()
        main_cfg.sql_confidence_threshold = 0.7

        verdict = ConfidenceResult(score=0.9, reasons=[], checks={})
        with patch(
            "openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg
        ), patch("openchatbi.text2sql.generate_sql.config.get", return_value=main_cfg), patch(
            "openchatbi.text2sql.generate_sql.SimpleSQLEvaluator"
        ) as MockEval:
            MockEval.return_value.evaluate.return_value = verdict
            out = execute_sql_node(_success_state())

        assert out["sql_execution_result"] == SQL_SUCCESS
        store.add.assert_called_once()
        args, kwargs = store.add.call_args
        assert args[0] == "how many"  # question
        assert kwargs["source"] == "auto"
        assert kwargs["namespace"] == "global"

    def test_capture_skipped_when_gate_fails(self, mock_llm, mock_catalog):
        """Confidence below threshold -> do not poison the example pool (success != correct)."""
        store = MagicMock()
        _gen, execute_sql_node, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.pattern_scope = "global"
        main_cfg = MagicMock()
        main_cfg.sql_confidence_threshold = 0.7

        verdict = ConfidenceResult(score=0.3, reasons=["WHERE missing"], checks={})
        with patch(
            "openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg
        ), patch("openchatbi.text2sql.generate_sql.config.get", return_value=main_cfg), patch(
            "openchatbi.text2sql.generate_sql.SimpleSQLEvaluator"
        ) as MockEval:
            MockEval.return_value.evaluate.return_value = verdict
            out = execute_sql_node(_success_state())

        assert out["sql_execution_result"] == SQL_SUCCESS
        store.add.assert_not_called()

    def test_capture_noop_on_unapproved_edit_reentry(self, mock_llm, mock_catalog):
        """Convention #10: edit re-entry (HITL, not yet approved) must not capture."""
        store = MagicMock()
        _gen, execute_sql_node, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.pattern_scope = "global"
        main_cfg = MagicMock()
        main_cfg.sql_confidence_threshold = 0.7

        verdict = ConfidenceResult(score=0.95, reasons=[], checks={})
        state = _success_state()
        state["human_sql_decision"] = "edit"  # pre-approval edit re-entry
        with patch(
            "openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg
        ), patch("openchatbi.text2sql.generate_sql.config.get", return_value=main_cfg), patch(
            "openchatbi.text2sql.generate_sql.SimpleSQLEvaluator"
        ) as MockEval:
            MockEval.return_value.evaluate.return_value = verdict
            out = execute_sql_node(state)

        assert out["sql_execution_result"] == SQL_SUCCESS
        store.add.assert_not_called()

    def test_capture_fires_on_approved_reentry(self, mock_llm, mock_catalog):
        """human_sql_decision == 'approve' on re-entry still captures (terminal success)."""
        store = MagicMock()
        _gen, execute_sql_node, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.pattern_scope = "global"
        main_cfg = MagicMock()
        main_cfg.sql_confidence_threshold = 0.7

        verdict = ConfidenceResult(score=0.95, reasons=[], checks={})
        state = _success_state()
        state["human_sql_decision"] = "approve"
        with patch(
            "openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg
        ), patch("openchatbi.text2sql.generate_sql.config.get", return_value=main_cfg), patch(
            "openchatbi.text2sql.generate_sql.SimpleSQLEvaluator"
        ) as MockEval:
            MockEval.return_value.evaluate.return_value = verdict
            out = execute_sql_node(state)

        assert out["sql_execution_result"] == SQL_SUCCESS
        store.add.assert_called_once()

    def test_capture_never_raises_on_store_error(self, mock_llm, mock_catalog):
        """Fire-and-forget: a store.add exception must not break the success response."""
        store = MagicMock()
        store.add.side_effect = RuntimeError("boom")
        _gen, execute_sql_node, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.pattern_scope = "global"
        main_cfg = MagicMock()
        main_cfg.sql_confidence_threshold = 0.7
        verdict = ConfidenceResult(score=0.9, reasons=[], checks={})
        with patch(
            "openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg
        ), patch("openchatbi.text2sql.generate_sql.config.get", return_value=main_cfg), patch(
            "openchatbi.text2sql.generate_sql.SimpleSQLEvaluator"
        ) as MockEval:
            MockEval.return_value.evaluate.return_value = verdict
            out = execute_sql_node(_success_state())
        assert out["sql_execution_result"] == SQL_SUCCESS


class TestBlendedRetrieval:
    def test_blended_retrieval_uses_store_when_enabled(self, mock_llm, mock_catalog):
        """generate_sql_node blends via store.retrieve with the composite_score adapter."""
        store = MagicMock()
        store.retrieve.return_value = [
            ("how many users", "SELECT COUNT(*) FROM test_table", ["test_table"])
        ]
        gen, _exec, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.max_patterns_per_query = 5
        state = SQLGraphState(
            messages=[],
            rewrite_question="how many users",
            tables=[{"table": "test_table", "columns": []}],
        )
        with patch("openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg):
            gen(state)

        store.retrieve.assert_called_once()
        _args, kwargs = store.retrieve.call_args
        assert kwargs.get("score_fn") is not None  # composite_score adapter injected

    def test_score_fn_adapter_signature_and_value(self, mock_llm, mock_catalog):
        """The injected score_fn must accept (metadata, base_rank) and return a float."""
        store = MagicMock()
        store.retrieve.return_value = []
        captured = {}

        def _retrieve(question, k=10, score_fn=None):
            captured["score_fn"] = score_fn
            return []

        store.retrieve.side_effect = _retrieve
        gen, _exec, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.max_patterns_per_query = 5
        mem_cfg.importance_decay_half_life_days = 30.0
        state = SQLGraphState(
            messages=[],
            rewrite_question="q",
            tables=[{"table": "test_table", "columns": []}],
        )
        with patch("openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg):
            gen(state)

        score_fn = captured["score_fn"]
        meta = {"importance": 2.0, "last_used": "", "use_count": 3}
        val = score_fn(meta, 0)  # (metadata, base_rank)
        assert isinstance(val, float)
        assert val > 0.0

    def test_blended_retrieval_respects_table_filter(self, mock_llm, mock_catalog):
        """Soft filter: patterns touching no selected table are dropped; matches kept."""
        captured = {}

        def _capture(messages):
            captured["prompt"] = messages[0].content  # SystemMessage holds the examples
            return AIMessage(content="SELECT 1")

        mock_llm.invoke.side_effect = _capture
        store = MagicMock()
        store.retrieve.return_value = [
            ("matches selected", "SELECT 1 FROM test_table", ["test_table"]),
            ("other tables", "SELECT 1 FROM unrelated", ["unrelated"]),
        ]
        gen, _exec, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = True
        mem_cfg.max_patterns_per_query = 5
        state = SQLGraphState(
            messages=[],
            rewrite_question="q",
            tables=[{"table": "test_table", "columns": []}],
        )
        with patch("openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg):
            gen(state)

        prompt = captured["prompt"]
        assert "matches selected" in prompt
        assert "other tables" not in prompt

    def test_legacy_static_path_when_store_disabled(self, mock_llm, mock_catalog):
        """Flag off -> legacy static retriever path (store.retrieve never called)."""
        captured = {}

        def _capture(messages):
            captured["prompt"] = messages[0].content
            return AIMessage(content="SELECT 1")

        mock_llm.invoke.side_effect = _capture
        store = MagicMock()
        gen, _exec, _regen, _viz, _score, _gate = create_sql_nodes(
            mock_llm, mock_catalog, "presto", learned_sql_store=store
        )
        mem_cfg = MagicMock()
        mem_cfg.enable_pattern_memory = False
        state = SQLGraphState(
            messages=[],
            rewrite_question="q",
            tables=[{"table": "test_table", "columns": []}],
        )
        with patch("openchatbi.text2sql.generate_sql.get_memory_config", return_value=mem_cfg), patch(
            "openchatbi.text2sql.generate_sql.sql_example_retriever"
        ) as mock_retriever:
            mock_retriever.invoke.return_value = []
            gen(state)

        store.retrieve.assert_not_called()
        mock_retriever.invoke.assert_called_once()
