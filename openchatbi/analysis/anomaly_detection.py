"""Core algorithms for time series anomaly detection.

The scoring strategy follows a rule/strategy-based design where each factor is
intentionally orthogonal so that they do not double-count the same evidence:

- Deviation significance: how statistically far the actual value is from the
  forecast, expressed in robust noise-scale (sigma) units. This is the single
  "how abnormal" signal and replaces the previously redundant
  reconstruction-error + bound-violation pair.
- Direction: drops vs rises can be weighted differently because, for most
  business metrics, an unexpected drop is more severe than a rise.
- Volume modulation: anomalies on high-traffic moments matter more than the
  same relative deviation on low-traffic moments. This is a multiplier driven
  by the expected (predicted) level, NOT by the anomalous actual value, so a
  drop-to-zero is never penalised by it.
- Historical anomaly frequency: noisy/jumpy metrics that violate their own
  bounds frequently are dampened, reducing false positives.
- Duration: a run of consecutive anomalous points near the end of the window
  boosts the score.

The final score is in the range [0, 1]; values closer to 1 indicate a more
significant / higher-impact anomaly.
"""

import math
from typing import Any

from openchatbi import config
from openchatbi.tool.timeseries_forecast import call_timeseries_service, check_forecast_service_health

# Number of sigma at which a deviation is considered to reach the "bound".
SIGMA_BOUND = 3.0
# Threshold on per-point severity above which a point is counted as anomalous.
ANOMALY_THRESHOLD = 0.6
# Default direction weights (neutral). Increase DROP_WEIGHT to emphasise drops.
DEFAULT_DROP_WEIGHT = 1.0
DEFAULT_RISE_WEIGHT = 1.0


def _extract_values(data: list[float | int | dict[str, Any]], target_column: str = "value") -> list[float]:
    """Extract numeric values from input data, silently skipping invalid entries."""
    values: list[float] = []
    for item in data:
        if isinstance(item, bool):
            # bool is a subclass of int; ignore to avoid treating True/False as data
            continue
        if isinstance(item, int | float):
            values.append(float(item))
        elif isinstance(item, dict) and target_column in item:
            try:
                values.append(float(item[target_column]))
            except (ValueError, TypeError):
                continue
    return values


def _mean(values: list[float]) -> float:
    """Arithmetic mean; returns 0.0 for empty input."""
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    """Median; returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _estimate_noise_scale(historical_values: list[float]) -> float:
    """Estimate the robust noise scale (sigma) of a series.

    Works on first differences (which removes trend/seasonality, so the scale is
    not inflated for periodic metrics the way the raw level std would be) and
    uses a MAD-based estimator so that occasional historical spikes do not
    inflate the scale. The standard deviation is only used as a fallback when
    the MAD is degenerate (e.g. mostly-constant series with a single jump).
    """
    if len(historical_values) < 2:
        return 0.0

    diffs = [historical_values[i] - historical_values[i - 1] for i in range(1, len(historical_values))]
    if len(diffs) < 2:
        return abs(diffs[0]) / math.sqrt(2)

    median_diff = _median(diffs)
    mad = _median([abs(d - median_diff) for d in diffs])
    # 1.4826 scales MAD to be consistent with std for normal data.
    sigma_diff = 1.4826 * mad
    if sigma_diff <= 0:
        mean_diff = _mean(diffs)
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        sigma_diff = math.sqrt(variance)
    return sigma_diff / math.sqrt(2)


def _deviation_significance(actual: float, predicted: float, sigma: float) -> tuple[float, float]:
    """Return (significance in [0, 1], signed z-score).

    The significance grows linearly to 0.5 at the SIGMA_BOUND (3 sigma) bound,
    then linearly to 1.0 at twice the bound (6 sigma), where it saturates.
    """
    # Floor sigma to a small fraction of the level so a near-constant series
    # still reacts to clear jumps without dividing by ~0.
    effective_sigma = max(sigma, abs(predicted) * 1e-3, 1e-9)
    z = (actual - predicted) / effective_sigma
    abs_z = abs(z)

    if abs_z <= SIGMA_BOUND:
        significance = 0.5 * (abs_z / SIGMA_BOUND)
    else:
        significance = 0.5 + 0.5 * min((abs_z - SIGMA_BOUND) / SIGMA_BOUND, 1.0)

    return significance, z


def _volume_factor(expected_level: float, historical_values: list[float]) -> float:
    """Modulation multiplier in [0.6, 1.0] based on the expected traffic level.

    High-traffic moments keep full weight (1.0); low-traffic moments are damped
    towards 0.6. Driven by the predicted/expected level so that the anomalous
    actual value (e.g. a drop to zero) never reduces the factor.
    """
    hist_mean = _mean(historical_values)
    if hist_mean <= 0:
        return 1.0
    relative_level = max(expected_level, 0.0) / hist_mean
    return 0.6 + 0.4 * min(relative_level, 1.0)


def _historical_anomaly_frequency(historical_values: list[float], sigma: float) -> float:
    """Fraction of historical steps whose jump exceeds the 3-sigma noise band.

    A high frequency means the metric is intrinsically noisy/jumpy, so anomalies
    are less alarming and the final score should be dampened.
    """
    if len(historical_values) < 3 or sigma <= 0:
        return 0.0

    diffs = [historical_values[i] - historical_values[i - 1] for i in range(1, len(historical_values))]
    center = _median(diffs)
    # sigma is the per-step noise; the diff series has std sigma * sqrt(2).
    threshold = SIGMA_BOUND * sigma * math.sqrt(2)
    if threshold <= 0:
        return 0.0
    violations = sum(1 for d in diffs if abs(d - center) > threshold)
    return violations / len(diffs)


def _evaluate_window(
    actual_values: list[float],
    predicted_values: list[float],
    historical_values: list[float],
    sigmas: list[float] | None = None,
    drop_weight: float = DEFAULT_DROP_WEIGHT,
    rise_weight: float = DEFAULT_RISE_WEIGHT,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a window of recent points and return (score in [0, 1], details).

    Newer points carry higher weight (linear weighting, gentler than exponential
    so the noisiest far-horizon forecast does not dominate). Each point combines
    an orthogonal set of factors; window-level frequency dampening and duration
    boosting are applied afterwards.
    """
    if not actual_values or not predicted_values:
        return 0.0, {"error": "Empty values provided for evaluation"}

    window_size = min(len(actual_values), len(predicted_values))

    # Global noise scale used when per-point sigmas are unavailable.
    global_sigma = _estimate_noise_scale(historical_values)
    anomaly_frequency = _historical_anomaly_frequency(historical_values, global_sigma)
    frequency_dampener = 1.0 - 0.4 * min(anomaly_frequency, 1.0)

    # Linear weights: oldest -> newest = 1 -> window_size.
    weights = [float(i + 1) for i in range(window_size)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    weighted_base = 0.0
    point_details: list[dict[str, Any]] = []
    consecutive_anomalies = 0

    for i in range(window_size):
        actual = actual_values[i]
        predicted = predicted_values[i]
        weight = normalized_weights[i]
        sigma_i = sigmas[i] if sigmas is not None and i < len(sigmas) else global_sigma

        significance, z = _deviation_significance(actual, predicted, sigma_i)

        direction = "drop" if z < 0 else "rise"
        direction_weight = drop_weight if z < 0 else rise_weight
        severity = min(significance * direction_weight, 1.0)

        volume_factor = _volume_factor(predicted, historical_values)
        contribution = severity * volume_factor

        if severity > ANOMALY_THRESHOLD:
            consecutive_anomalies += 1
        else:
            consecutive_anomalies = 0

        weighted_base += contribution * weight

        point_details.append(
            {
                "actual": actual,
                "predicted": predicted,
                "z_score": z,
                "significance": significance,
                "direction": direction,
                "volume_factor": volume_factor,
                "severity": severity,
                "contribution": contribution,
                "weight": weight,
            }
        )

    # Duration factor: boost when several consecutive points are anomalous.
    duration_boost = 0.0
    if consecutive_anomalies > 1:
        duration_boost = min((consecutive_anomalies - 1) * 0.1, 0.2)

    final_score = max(0.0, min(weighted_base * frequency_dampener + duration_boost, 1.0))

    details = {
        "window_size": window_size,
        "noise_scale": global_sigma,
        "historical_anomaly_frequency": anomaly_frequency,
        "frequency_dampener": frequency_dampener,
        "weighted_base": weighted_base,
        "consecutive_anomalies": consecutive_anomalies,
        "duration_boost": duration_boost,
        "point_details": point_details,
    }

    return final_score, details


def evaluate_anomalies(
    input_data: list[float | int | dict[str, Any]],
    evaluation_window: int = 3,
    frequency: str = "hourly",
    target_column: str = "value",
    input_length: int | None = None,
    drop_weight: float = DEFAULT_DROP_WEIGHT,
    rise_weight: float = DEFAULT_RISE_WEIGHT,
) -> tuple[float, dict[str, Any]]:
    """Forecast the evaluation window and score it for anomalies.

    The last ``evaluation_window`` points of ``input_data`` are treated as the
    points to evaluate; the preceding points are used both as the forecast input
    and as the historical context for the noise scale, volume and frequency
    factors.

    Args:
        input_data: Time series as numbers or dicts; evaluation points are at the end.
        evaluation_window: Number of trailing points to evaluate (must be < len(input_data)).
        frequency: Time series frequency (e.g. 'hourly', 'daily').
        target_column: Column to read from structured (dict) data.
        input_length: Optional cap on how much history to send to the forecast service.
        drop_weight: Severity multiplier applied to downward deviations.
        rise_weight: Severity multiplier applied to upward deviations.

    Returns:
        Tuple of (anomaly score in [0, 1], details dict). On any failure the
        score is 0.0 and details contains an ``"error"`` key.
    """
    if not input_data or len(input_data) <= evaluation_window:
        return 0.0, {
            "error": (
                f"Input data length ({len(input_data)}) must be greater than "
                f"evaluation window ({evaluation_window})."
            )
        }

    # Check service availability before doing any work.
    if not check_forecast_service_health():
        return 0.0, {"error": "Time Series Forecasting Service Unavailable. Cannot perform anomaly detection."}

    service_url = config.get().timeseries_forecasting_service_url

    # Split data into historical context and the evaluation window.
    historical_data = input_data[:-evaluation_window]
    evaluation_data = input_data[-evaluation_window:]

    hist_values = _extract_values(historical_data, target_column)
    eval_values = _extract_values(evaluation_data, target_column)

    if not hist_values or not eval_values:
        return 0.0, {"error": "Could not extract numeric values from input data."}

    # Forecast the evaluation window using only the historical context.
    forecast_result = call_timeseries_service(
        service_url=service_url,
        input_data=historical_data,
        forecast_window=evaluation_window,
        frequency=frequency,
        input_length=input_length,
        target_column=target_column,
    )

    if forecast_result.get("status") in ("error", "http_error"):
        return 0.0, {
            "error": f"Forecasting Error during anomaly detection: {forecast_result.get('error', 'Unknown error')}"
        }

    predicted_values = forecast_result.get("predictions", [])

    if not predicted_values or len(predicted_values) != len(eval_values):
        return 0.0, {
            "error": (
                f"Mismatch between predicted values ({len(predicted_values)}) "
                f"and evaluation window size ({len(eval_values)})."
            )
        }

    # Forward-compatible: if the forecast service ever returns per-horizon
    # uncertainty, prefer it over the historical noise-scale estimate.
    sigmas = forecast_result.get("prediction_std")
    if not isinstance(sigmas, list) or len(sigmas) != len(predicted_values):
        sigmas = None

    return _evaluate_window(
        actual_values=eval_values,
        predicted_values=predicted_values,
        historical_values=hist_values,
        sigmas=sigmas,
        drop_weight=drop_weight,
        rise_weight=rise_weight,
    )


def format_anomaly_report(score: float, details: dict[str, Any]) -> str:
    """Format the anomaly detection result into a human-readable report."""
    if "error" in details:
        return f"Anomaly Detection Error: {details['error']}"

    severity = "Low"
    if score > 0.8:
        severity = "Critical"
    elif score > 0.6:
        severity = "High"
    elif score > 0.4:
        severity = "Medium"

    report = [
        "🚨 Anomaly Detection Report",
        f"Overall Anomaly Score: {score:.4f} (Severity: {severity})",
        "",
        "Evaluation Summary:",
        f"  • Window Size: {details.get('window_size', 0)} points",
        f"  • Noise Scale (sigma): {details.get('noise_scale', 0.0):.4f}",
        f"  • Historical Anomaly Frequency: {details.get('historical_anomaly_frequency', 0.0):.4f}",
        f"  • Frequency Dampener: x{details.get('frequency_dampener', 1.0):.4f}",
        f"  • Consecutive Anomalies: {details.get('consecutive_anomalies', 0)}",
        f"  • Duration Boost Applied: +{details.get('duration_boost', 0.0):.4f}",
        "",
        "Detailed Point Evaluation (Oldest to Newest):",
    ]

    for i, point in enumerate(details.get("point_details", [])):
        report.extend(
            [
                f"  Point {i + 1} (Weight: {point['weight']:.2f}):",
                f"    - Actual: {point['actual']:.2f} | Predicted: {point['predicted']:.2f} " f"({point['direction']})",
                f"    - Z-Score: {point['z_score']:.2f} | Significance: {point['significance']:.4f}",
                f"    - Volume Factor: x{point['volume_factor']:.4f}",
                f"    - Point Severity: {point['severity']:.4f} | Contribution: {point['contribution']:.4f}",
            ]
        )

    if score > 0.6:
        report.extend(
            [
                "",
                "💡 Insight: Significant anomaly detected. The actual values deviate "
                "substantially from the expected pattern.",
            ]
        )
    else:
        report.extend(
            [
                "",
                "💡 Insight: No significant anomaly detected. The data follows the expected pattern closely.",
            ]
        )

    return "\n".join(report)
