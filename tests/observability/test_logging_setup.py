"""Tests for opt-in JSON logging setup."""

import json
import logging

from openchatbi.observability.context import set_run_context
from openchatbi.observability.logging_setup import setup_logging


def test_setup_logging_does_not_clobber_existing_handlers() -> None:
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    try:
        setup_logging(level="INFO", json=True)
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)
        for h in list(root.handlers):
            if getattr(h, "_openchatbi_obs", False):
                root.removeHandler(h)


def test_setup_logging_emits_json_with_context_fields(capsys) -> None:
    set_run_context("bob", "req-9")
    setup_logging(level="INFO", json=True)
    logging.getLogger("openchatbi.test").info("hello")
    err = capsys.readouterr().err
    line = [ln for ln in err.splitlines() if "hello" in ln][-1]
    payload = json.loads(line)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["user_id"] == "bob"
    assert payload["request_id"] == "req-9"
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_openchatbi_obs", False):
            root.removeHandler(h)
