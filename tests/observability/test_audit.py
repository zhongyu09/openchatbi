"""Tests for the structured audit logger + SQL/arg masking."""

from openchatbi.observability.audit import AuditLogger, mask_args, mask_sql


def test_mask_sql_redacts_string_and_number_literals() -> None:
    masked = mask_sql("SELECT * FROM users WHERE name = 'alice' AND age = 42")
    assert "alice" not in masked
    assert "42" not in masked
    assert "SELECT" in masked and "FROM users" in masked


def test_mask_args_redacts_values_keeps_keys() -> None:
    out = mask_args({"question": "who is alice", "token": "secret123"})
    assert set(out.keys()) == {"question", "token"}
    assert out["token"] != "secret123"


def test_log_sql_exec_emits_structured_record(caplog) -> None:
    import logging

    logger = AuditLogger()
    with caplog.at_level(logging.INFO, logger="openchatbi.audit"):
        logger.log_sql_exec(
            sql="SELECT COUNT(*) FROM users WHERE id = 7",
            dialect="presto",
            row_count=1,
            duration_ms=12.5,
            status="SQL_SUCCESS",
            user_id="alice",
        )
    assert any("SQL_SUCCESS" in r.message for r in caplog.records)
    # Raw literal must never reach the audit sink.
    assert all("= 7" not in r.message for r in caplog.records)
