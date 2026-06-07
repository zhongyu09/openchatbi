"""Tests for the shared forecasting service caller and min-input-length resolution."""

import openchatbi.tool.timeseries_forecast as tf
from openchatbi.tool.timeseries_forecast import (
    DEFAULT_MIN_FORECAST_INPUT_LENGTH,
    call_timeseries_service,
    get_min_forecast_input_length,
)


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    @property
    def text(self):
        return str(self._payload)


def _patch_post(monkeypatch, captured):
    def _post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp(200, {"predictions": [1.0], "status": "success"})

    monkeypatch.setattr(tf.requests, "post", _post)


# --- call_timeseries_service padding decision (min length stubbed) ---


def test_sets_input_len_when_history_below_minimum(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, captured)
    monkeypatch.setattr(tf, "get_min_forecast_input_length", lambda url: 96)

    call_timeseries_service("http://fake", [1.0, 2.0, 3.0], forecast_window=1, frequency="daily")

    assert captured["payload"]["input_len"] == 96


def test_does_not_set_input_len_when_history_sufficient(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, captured)
    monkeypatch.setattr(tf, "get_min_forecast_input_length", lambda url: 96)

    long_input = [float(i) for i in range(96)]
    call_timeseries_service("http://fake", long_input, forecast_window=1, frequency="daily")

    assert "input_len" not in captured["payload"]


def test_respects_explicit_input_len(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, captured)

    call_timeseries_service("http://fake", [1.0, 2.0, 3.0], forecast_window=1, frequency="daily", input_length=200)

    assert captured["payload"]["input_len"] == 200


# --- get_min_forecast_input_length resolution ---


def test_min_length_from_service_health(monkeypatch):
    tf._min_input_length_cache.clear()
    monkeypatch.setattr(tf, "_configured_min_input_length", lambda: None)
    monkeypatch.setattr(tf.requests, "get", lambda url, timeout=None: _Resp(200, {"min_input_length": 48}))

    assert get_min_forecast_input_length("http://svc") == 48
    # value is cached for the service URL
    assert tf._min_input_length_cache["http://svc"] == 48


def test_min_length_config_override_wins(monkeypatch):
    tf._min_input_length_cache.clear()
    monkeypatch.setattr(tf, "_configured_min_input_length", lambda: 64)

    def _should_not_fetch(url, timeout=None):
        raise AssertionError("override should short-circuit the health fetch")

    monkeypatch.setattr(tf.requests, "get", _should_not_fetch)

    assert get_min_forecast_input_length("http://svc") == 64


def test_min_length_fallback_when_service_unreachable(monkeypatch):
    tf._min_input_length_cache.clear()
    monkeypatch.setattr(tf, "_configured_min_input_length", lambda: None)

    def _raise(url, timeout=None):
        raise tf.requests.exceptions.RequestException("boom")

    monkeypatch.setattr(tf.requests, "get", _raise)

    assert get_min_forecast_input_length("http://svc") == DEFAULT_MIN_FORECAST_INPUT_LENGTH


def test_min_length_fallback_when_field_missing(monkeypatch):
    tf._min_input_length_cache.clear()
    monkeypatch.setattr(tf, "_configured_min_input_length", lambda: None)
    monkeypatch.setattr(tf.requests, "get", lambda url, timeout=None: _Resp(200, {"status": "healthy"}))

    assert get_min_forecast_input_length("http://svc") == DEFAULT_MIN_FORECAST_INPUT_LENGTH


def test_health_check_warms_min_length_cache(monkeypatch):
    # The health check should prime the cache from the same /health payload, so a subsequent
    # get_min_forecast_input_length resolves without a second request.
    tf._min_input_length_cache.clear()
    monkeypatch.setattr(
        tf.requests,
        "get",
        lambda url, timeout=None: _Resp(200, {"model_initialized": True, "min_input_length": 48}),
    )

    assert tf._probe_service_health("http://svc") is True
    assert tf._min_input_length_cache["http://svc"] == 48

    # Subsequent resolution must hit the cache and NOT fetch again.
    monkeypatch.setattr(tf, "_configured_min_input_length", lambda: None)

    def _should_not_fetch(url, timeout=None):
        raise AssertionError("cache should be used; no second /health fetch expected")

    monkeypatch.setattr(tf.requests, "get", _should_not_fetch)
    assert get_min_forecast_input_length("http://svc") == 48
