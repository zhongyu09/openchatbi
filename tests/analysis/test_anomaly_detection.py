import pytest

from openchatbi.analysis import anomaly_detection as ad
from openchatbi.analysis.anomaly_detection import (
    _deviation_significance,
    _estimate_noise_scale,
    _evaluate_window,
    _extract_values,
    _historical_anomaly_frequency,
    _volume_factor,
    evaluate_anomalies,
    format_anomaly_report,
)


# Smooth, mildly noisy history with mean ~ 1000.
SMOOTH_HISTORY = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1002.0, 998.0, 1008.0, 992.0, 1004.0] * 3


def test_extract_values():
    assert _extract_values([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert _extract_values([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert _extract_values([{"value": 10}, {"value": 20}]) == [10.0, 20.0]
    assert _extract_values([{"custom": 10}, {"custom": 20}], target_column="custom") == [10.0, 20.0]

    # Boundary cases
    assert _extract_values([]) == []
    assert _extract_values([{"wrong_col": 10}]) == []
    assert _extract_values([{"value": "not_a_number"}, {"value": 20}]) == [20.0]
    assert _extract_values([None, "string", 10.0]) == [10.0]
    # bool must not be treated as numeric data
    assert _extract_values([True, False, 5]) == [5.0]


def test_estimate_noise_scale():
    # Constant series has no noise.
    assert _estimate_noise_scale([100.0, 100.0, 100.0]) == 0.0
    # Too short to estimate.
    assert _estimate_noise_scale([]) == 0.0
    assert _estimate_noise_scale([100.0]) == 0.0
    # A series with variation produces a positive scale.
    assert _estimate_noise_scale(SMOOTH_HISTORY) > 0.0


def test_deviation_significance():
    # No deviation.
    sig, z = _deviation_significance(100.0, 100.0, 10.0)
    assert sig == 0.0
    assert z == 0.0
    # Exactly at the 3-sigma bound -> 0.5.
    sig, z = _deviation_significance(130.0, 100.0, 10.0)
    assert sig == pytest.approx(0.5)
    assert z == pytest.approx(3.0)
    # At/beyond 6 sigma -> saturates at 1.0.
    sig, _ = _deviation_significance(160.0, 100.0, 10.0)
    assert sig == pytest.approx(1.0)
    # Direction is captured by the sign of z.
    _, z_down = _deviation_significance(70.0, 100.0, 10.0)
    assert z_down < 0
    # Zero sigma is floored so a clear jump on a flat series is still detected.
    sig, _ = _deviation_significance(101.0, 100.0, 0.0)
    assert sig > 0.5


def test_volume_factor():
    # Empty history -> neutral.
    assert _volume_factor(1000.0, []) == 1.0
    # Non-positive mean -> neutral.
    assert _volume_factor(1000.0, [0.0, 0.0]) == 1.0
    # Level above the historical mean -> full weight.
    assert _volume_factor(1500.0, SMOOTH_HISTORY) == pytest.approx(1.0)
    # Low-traffic moment is damped towards 0.6.
    assert _volume_factor(0.0, SMOOTH_HISTORY) == pytest.approx(0.6)
    # Mid-level is between the two extremes.
    assert 0.6 < _volume_factor(500.0, SMOOTH_HISTORY) < 0.9


def test_historical_anomaly_frequency():
    # Smooth history -> ~no historical anomalies.
    assert _historical_anomaly_frequency(SMOOTH_HISTORY, _estimate_noise_scale(SMOOTH_HISTORY)) < 0.1
    # Degenerate inputs.
    assert _historical_anomaly_frequency([1.0, 2.0], 1.0) == 0.0
    assert _historical_anomaly_frequency(SMOOTH_HISTORY, 0.0) == 0.0


def test_historical_anomaly_frequency_detects_spikes():
    # Mostly smooth series with a few isolated spikes: the robust noise scale
    # stays small, so the spikes are counted as historical anomalies.
    spiky = [1000.0 + (5.0 if i % 2 else -5.0) for i in range(30)]
    for idx in (7, 15, 23):
        spiky[idx] = 1600.0
    freq = _historical_anomaly_frequency(spiky, _estimate_noise_scale(spiky))
    assert freq > 0.0


def test_evaluate_window_normal():
    actual = [1000.0, 1010.0, 990.0]
    predicted = [1000.0, 1000.0, 1000.0]
    score, details = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert score < 0.3
    assert details["consecutive_anomalies"] == 0


def test_evaluate_window_legit_growth_not_flagged():
    # A large but expected value (actual matches forecast) must NOT be flagged,
    # even though its absolute magnitude is high.
    actual = [2000.0, 2000.0, 2000.0]
    predicted = [2000.0, 2000.0, 2000.0]
    score, _ = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert score < 0.1


def test_evaluate_window_drop_to_zero_high_traffic():
    # A drop to zero on a high-traffic metric is the most severe anomaly and
    # must score near 1 (regression test for the relative-size sign bug).
    actual = [1000.0, 500.0, 0.0]
    predicted = [1000.0, 1000.0, 1000.0]
    score, details = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert score > 0.8
    assert details["point_details"][-1]["direction"] == "drop"


def test_evaluate_window_direction_weighting():
    # Use a mild deviation that stays inside the linear (<3 sigma) band so the
    # direction weight is not masked by saturation. SMOOTH_HISTORY sigma ~ 10.
    predicted = [1000.0, 1000.0, 1000.0]
    drop_actual = [1000.0, 1000.0, 985.0]
    rise_actual = [1000.0, 1000.0, 1015.0]

    drop_default, _ = _evaluate_window(drop_actual, predicted, SMOOTH_HISTORY)
    drop_emphasised, _ = _evaluate_window(
        drop_actual, predicted, SMOOTH_HISTORY, drop_weight=2.0, rise_weight=1.0
    )
    rise_default, _ = _evaluate_window(rise_actual, predicted, SMOOTH_HISTORY)

    # Emphasising drops raises the score for a downward deviation.
    assert drop_emphasised > drop_default
    # With neutral weights, a symmetric drop and rise score the same.
    assert drop_default == pytest.approx(rise_default)


def test_evaluate_window_noisy_history_dampens():
    # The same absolute deviation is far less significant on a high-variance
    # series (large noise scale) than on a smooth one, so it scores lower.
    actual = [1000.0, 1000.0, 1500.0]
    predicted = [1000.0, 1000.0, 1000.0]
    noisy_history = [1000.0, 100.0, 1900.0, 200.0, 1800.0, 50.0, 1950.0, 150.0] * 3

    smooth_score, _ = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    noisy_score, _ = _evaluate_window(actual, predicted, noisy_history)
    assert noisy_score < smooth_score


def test_evaluate_window_duration_boost():
    actual = [400.0, 350.0, 300.0]
    predicted = [1000.0, 1000.0, 1000.0]
    score, details = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert details["consecutive_anomalies"] == 3
    assert details["duration_boost"] > 0.0
    assert score > 0.8


def test_evaluate_window_boundary_cases():
    # Empty inputs.
    score, details = _evaluate_window([], [], SMOOTH_HISTORY)
    assert score == 0.0
    assert "error" in details
    # Mismatched lengths -> evaluate the overlap only.
    _, details = _evaluate_window([100.0], [100.0, 100.0], SMOOTH_HISTORY)
    assert details["window_size"] == 1
    # Per-point sigmas are honoured when supplied.
    score, _ = _evaluate_window([100.0], [100.0], SMOOTH_HISTORY, sigmas=[1.0])
    assert score == 0.0


def _patch_service(monkeypatch, predictions, status="success", healthy=True):
    """Patch the forecast service dependencies used by evaluate_anomalies."""
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: healthy)

    class _FakeCfg:
        timeseries_forecasting_service_url = "http://fake"

    monkeypatch.setattr(ad.config, "get", lambda: _FakeCfg())

    def _fake_call(**kwargs):
        if status != "success":
            return {"status": status, "error": "boom"}
        return {"predictions": predictions, "status": "success"}

    monkeypatch.setattr(ad, "call_timeseries_service", _fake_call)


def test_evaluate_anomalies_success(monkeypatch):
    data = SMOOTH_HISTORY + [1000.0, 500.0, 0.0]
    _patch_service(monkeypatch, predictions=[1000.0, 1000.0, 1000.0])
    score, details = evaluate_anomalies(data, evaluation_window=3)
    assert "error" not in details
    assert score > 0.8


def test_evaluate_anomalies_input_too_short():
    score, details = evaluate_anomalies([1.0, 2.0], evaluation_window=3)
    assert score == 0.0
    assert "error" in details


def test_evaluate_anomalies_service_unavailable(monkeypatch):
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: False)
    score, details = evaluate_anomalies(SMOOTH_HISTORY + [1.0, 2.0, 3.0], evaluation_window=3)
    assert score == 0.0
    assert "Unavailable" in details["error"]


def test_evaluate_anomalies_forecast_error(monkeypatch):
    _patch_service(monkeypatch, predictions=[], status="error")
    score, details = evaluate_anomalies(SMOOTH_HISTORY + [1.0, 2.0, 3.0], evaluation_window=3)
    assert score == 0.0
    assert "error" in details


def test_evaluate_anomalies_prediction_length_mismatch(monkeypatch):
    _patch_service(monkeypatch, predictions=[1000.0])  # only 1, need 3
    score, details = evaluate_anomalies(SMOOTH_HISTORY + [1.0, 2.0, 3.0], evaluation_window=3)
    assert score == 0.0
    assert "Mismatch" in details["error"]


def test_format_anomaly_report():
    score, details = _evaluate_window([1000.0, 500.0, 0.0], [1000.0, 1000.0, 1000.0], SMOOTH_HISTORY)
    report = format_anomaly_report(score, details, reasoning="check traffic drop")
    assert "Anomaly Detection Report" in report
    assert "Severity" in report
    # Error details render an error report.
    assert "Anomaly Detection Error" in format_anomaly_report(0.0, {"error": "boom"})
