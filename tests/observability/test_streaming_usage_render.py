"""CLI renderer prints 'Turn: N tokens (~$X)'; async_api serializes StreamUsage."""

from openchatbi.streaming import StreamUsage


def test_cli_renders_turn_usage(capsys) -> None:
    from run_cli import CliRenderer

    renderer = CliRenderer(as_json=False, color=False)
    renderer.render(StreamUsage(turn_tokens=120, turn_cost_usd=0.0012, by_model={"gpt-4o": 120}))
    out = capsys.readouterr().out
    assert "Turn: 120 tokens" in out
    assert "$0.0012" in out or "$0.001" in out


def test_cli_json_usage(capsys) -> None:
    import json

    from run_cli import CliRenderer

    renderer = CliRenderer(as_json=True, color=False)
    renderer.render(StreamUsage(turn_tokens=120, turn_cost_usd=0.0012, by_model={"gpt-4o": 120}))
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["type"] == "usage"
    assert payload["turn_tokens"] == 120


def test_api_serializes_usage() -> None:
    from sample_api.async_api import _event_to_dict

    d = _event_to_dict(StreamUsage(turn_tokens=120, turn_cost_usd=0.0012, by_model={"gpt-4o": 120}))
    assert d["type"] == "usage"
    assert d["turn_tokens"] == 120
    assert d["by_model"] == {"gpt-4o": 120}
