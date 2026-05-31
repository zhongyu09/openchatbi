import pytest
from openchatbi.analysis.anomaly_detection import (
    _calculate_reconstruction_error,
    _calculate_relative_size,
    _calculate_bound_violation,
    _evaluate_window,
    _extract_values,
)


def test_extract_values():
    # Test simple floats
    assert _extract_values([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    # Test ints
    assert _extract_values([1, 2, 3]) == [1.0, 2.0, 3.0]
    # Test dicts
    assert _extract_values([{"value": 10}, {"value": 20}]) == [10.0, 20.0]
    # Test custom column
    assert _extract_values([{"custom": 10}, {"custom": 20}], target_column="custom") == [10.0, 20.0]

    # Boundary cases
    assert _extract_values([]) == []
    assert _extract_values([{"wrong_col": 10}]) == []
    assert _extract_values([{"value": "not_a_number"}, {"value": 20}]) == [20.0]
    assert _extract_values([None, "string", 10.0]) == [10.0]


def test_calculate_reconstruction_error():
    # Perfect match
    assert _calculate_reconstruction_error(100.0, 100.0) == 0.0
    # 50% error
    assert _calculate_reconstruction_error(150.0, 100.0) == 0.5
    # Capped at 1.0
    assert _calculate_reconstruction_error(300.0, 100.0) == 1.0
    # Zero prediction handling
    assert _calculate_reconstruction_error(10.0, 0.0) == 1.0
    # Negative values
    assert _calculate_reconstruction_error(-50.0, -100.0) == 0.5
    assert _calculate_reconstruction_error(50.0, -50.0) == 1.0


def test_calculate_relative_size():
    history = [100.0, 100.0, 100.0]
    # Same as history
    assert 0.4 < _calculate_relative_size(100.0, history) < 0.6
    # Much larger
    assert _calculate_relative_size(1000.0, history) > 0.9
    # Much smaller
    assert _calculate_relative_size(10.0, history) < 0.5

    # Boundary cases
    assert _calculate_relative_size(100.0, []) == 0.5  # Empty history
    assert _calculate_relative_size(100.0, [0.0, 0.0]) > 0.9  # Zero history mean
    assert _calculate_relative_size(-100.0, history) < 0.5  # Negative actual


def test_calculate_bound_violation():
    history = [90.0, 100.0, 110.0]  # mean=100, std=10, 3*std=30
    # Inside bounds (predicted=100, bound=[70, 130])
    assert _calculate_bound_violation(100.0, 100.0, history) == 0.0
    assert _calculate_bound_violation(120.0, 100.0, history) == 0.0
    # Outside bounds
    assert _calculate_bound_violation(140.0, 100.0, history) > 0.5
    assert _calculate_bound_violation(160.0, 100.0, history) == 1.0

    # Boundary cases
    # Empty history fallback to 20% deviation
    assert _calculate_bound_violation(100.0, 100.0, []) == 0.0
    assert _calculate_bound_violation(130.0, 100.0, []) > 0.5
    # History with 1 element fallback to 20% deviation
    assert _calculate_bound_violation(100.0, 100.0, [100.0]) == 0.0
    assert _calculate_bound_violation(130.0, 100.0, [100.0]) > 0.5
    # Zero variance history
    assert _calculate_bound_violation(100.0, 100.0, [100.0, 100.0]) == 0.0
    assert _calculate_bound_violation(101.0, 100.0, [100.0, 100.0]) > 0.5


def test_evaluate_window():
    # Normal case
    actual = [100.0, 105.0, 95.0]
    predicted = [100.0, 100.0, 100.0]
    # Add some variance to history so std_dev is not 0
    history = [100.0, 105.0, 95.0, 102.0, 98.0] * 4

    score, details = _evaluate_window(actual, predicted, history)
    assert score < 0.3  # Should be low anomaly score
    assert details["consecutive_anomalies"] == 0

    # Anomaly case (spike)
    actual_anomaly = [100.0, 105.0, 300.0]
    score_anomaly, details_anomaly = _evaluate_window(actual_anomaly, predicted, history)
    assert score_anomaly > 0.5  # Should be higher due to the last point

    # Sustained anomaly
    actual_sustained = [250.0, 280.0, 300.0]
    score_sustained, details_sustained = _evaluate_window(actual_sustained, predicted, history)
    assert score_sustained > 0.8
    assert details_sustained["consecutive_anomalies"] == 3
    assert details_sustained["duration_boost"] > 0.0

    # Boundary cases
    # Empty inputs
    score_empty, details_empty = _evaluate_window([], [], history)
    assert score_empty == 0.0
    assert "error" in details_empty

    # Mismatched lengths
    score_mismatch, details_mismatch = _evaluate_window([100.0], [100.0, 100.0], history)
    assert details_mismatch["window_size"] == 1
