"""Opt-in structured (JSON) logging for the stdlib root logger.

Intentionally NOT called on import: embedding hosts keep their own logging.
``setup_logging`` only adds a handler when the root has none of ours, never
removes existing handlers, and injects run-context fields into every record.
"""

from __future__ import annotations

import json
import logging
import sys

from openchatbi.observability.context import get_run_context


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        user_id, request_id = get_run_context()
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "user_id": user_id,
            "request_id": request_id,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", json: bool = True) -> None:
    """Configure the root logger once (opt-in; never clobbers existing handlers)."""
    root = logging.getLogger()
    if any(getattr(h, "_openchatbi_obs", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler._openchatbi_obs = True  # type: ignore[attr-defined]
    if json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
