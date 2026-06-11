"""Config.observability submodel is declared (pydantic extra='ignore' would
otherwise silently drop it)."""

from unittest.mock import MagicMock

from openchatbi.config_loader import Config, ObservabilityConfig


def test_observability_defaults_off() -> None:
    cfg = Config.from_dict({"default_llm": MagicMock()})
    assert isinstance(cfg.observability, ObservabilityConfig)
    assert cfg.observability.tracing.enabled is False
    assert cfg.observability.metrics.enabled is False
    assert cfg.observability.audit.enabled is False


def test_observability_parsed_from_dict() -> None:
    cfg = Config.from_dict(
        {
            "default_llm": MagicMock(),
            "observability": {
                "tracing": {"enabled": True, "provider": "langfuse"},
                "metrics": {"enabled": True, "prometheus_port": 9100},
                "audit": {"enabled": True, "mask_sql_literals": True},
            },
        }
    )
    assert cfg.observability.tracing.enabled is True
    assert cfg.observability.tracing.provider == "langfuse"
    assert cfg.observability.metrics.prometheus_port == 9100
    assert cfg.observability.audit.mask_sql_literals is True
