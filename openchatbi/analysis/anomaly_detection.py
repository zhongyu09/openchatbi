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
# Lower-window smoothing strength in [0, 1]. The window score is attributed to
# the last point (anchor); this knob blends the anchor's own contribution with a
# newest-weighted average of the window. 0 = pure anchor (most sensitive),
# 1 = full smoothed average (most denoised). A light default keeps the anchor
# dominant while damping isolated single-point spikes.
DEFAULT_SMOOTHING = 0.5

# Interval-segmentation defaults for the upper (detection_range) layer.
# These are intentionally conservative starting points and are expected to be
# tuned against real normal/anomalous samples (see design.md "Open Questions").
# Hysteresis thresholds: open an interval when severity rises above T_HIGH,
# close it only when severity falls below T_LOW (T_LOW < T_HIGH).
DEFAULT_T_HIGH = ANOMALY_THRESHOLD
DEFAULT_T_LOW = 0.4
# Adjacent intervals separated by <= MERGE_GAP points are merged into one.
DEFAULT_MERGE_GAP = 1
# An interval is kept if its area (sum of severity) >= AREA_MIN OR its peak
# severity >= PEAK_HIGH (the latter is a high-peak override for sharp, short
# anomalies that would otherwise be dropped by a length-only filter).
DEFAULT_AREA_MIN = 1.0
DEFAULT_PEAK_HIGH = 0.8

# Candidate keys used to surface a human-readable timestamp from structured rows.
_TIMESTAMP_KEYS = ("timestamp", "time", "datetime", "date", "ts", "dt", "period")


def _default_evaluation_window(frequency: str) -> int:
    """Return the internal lower-window size for a given frequency.

    Finer granularities are noisier, so they use a slightly longer window to
    denoise; coarser granularities keep the short default.
    """
    freq = (frequency or "").lower()
    if "min" in freq:
        return 6
    return 3


def _default_stride(frequency: str) -> int:
    """Return the default block-forecast stride (= horizon) for a frequency.

    The stride bounds the multi-step forecast horizon per block: small enough to
    avoid far-horizon smoothing, large enough to keep the call count low. Coarse
    granularities (weekly/monthly) have few points, so they use a small stride.
    """
    freq = (frequency or "").lower()
    if "min" in freq:
        return 12
    if "hour" in freq:
        return 12
    if "day" in freq or freq == "daily":
        return 7
    if "week" in freq:
        return 4
    if "month" in freq:
        return 4
    return 10


def _extract_timestamp(item: Any) -> Any | None:
    """Return a timestamp-like value from a structured row, if present."""
    if isinstance(item, dict):
        for key in _TIMESTAMP_KEYS:
            if key in item:
                return item[key]
    return None


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
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a window of recent points and return (score in [0, 1], details).

    The window score is attributed to its LAST point (the anchor); the preceding
    points only provide context for light smoothing/denoising and duration
    corroboration. ``smoothing`` blends the anchor's own contribution with a
    newest-weighted (linear) average of the window: 0 keeps the raw anchor (most
    sensitive), 1 uses the full smoothed average (most denoised). Window-level
    frequency dampening and duration boosting are applied afterwards.
    """
    if not actual_values or not predicted_values:
        return 0.0, {"error": "Empty values provided for evaluation"}

    window_size = min(len(actual_values), len(predicted_values))

    # Global noise scale used when per-point sigmas are unavailable.
    global_sigma = _estimate_noise_scale(historical_values)
    anomaly_frequency = _historical_anomaly_frequency(historical_values, global_sigma)
    frequency_dampener = 1.0 - 0.4 * min(anomaly_frequency, 1.0)

    # Linear weights for the smoothing kernel: oldest -> newest = 1 -> window_size.
    weights = [float(i + 1) for i in range(window_size)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    contributions: list[float] = []
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

        contributions.append(contribution)

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

    # Anchor attribution: the score belongs to the last point. The preceding
    # points denoise it via a light, newest-weighted average controlled by
    # ``smoothing`` (anchor weight share = 1 - smoothing).
    anchor_contribution = contributions[-1]
    smoothed_contribution = sum(c * w for c, w in zip(contributions, normalized_weights))
    smoothing = min(max(smoothing, 0.0), 1.0)
    window_base = (1.0 - smoothing) * anchor_contribution + smoothing * smoothed_contribution

    # Duration factor: boost when several consecutive points are anomalous.
    duration_boost = 0.0
    if consecutive_anomalies > 1:
        duration_boost = min((consecutive_anomalies - 1) * 0.1, 0.2)

    final_score = max(0.0, min(window_base * frequency_dampener + duration_boost, 1.0))

    details = {
        "window_size": window_size,
        "noise_scale": global_sigma,
        "historical_anomaly_frequency": anomaly_frequency,
        "frequency_dampener": frequency_dampener,
        "weighted_base": window_base,
        "anchor_contribution": anchor_contribution,
        "smoothed_contribution": smoothed_contribution,
        "smoothing": smoothing,
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
    smoothing: float = DEFAULT_SMOOTHING,
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
        input_length: Target historical input length for the service; if the supplied history is
            shorter, the service left-pads the earliest points with zeros to reach this length.
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
        smoothing=smoothing,
    )


def _block_forecast(
    input_data: list[float | int | dict[str, Any]],
    scan_start: int,
    detection_range: int,
    stride: int,
    frequency: str,
    input_length: int | None,
    target_column: str,
    service_url: str,
) -> tuple[list[float | None], list[dict[str, Any]]]:
    """Forecast the scan range in non-overlapping blocks of size ``stride``.

    Each block is forecast from the history that precedes it (``input_data[:a]``),
    so the anomalous actual values inside a block never enter the model input.
    Returns a list of length ``detection_range`` (predictions aligned to the scan
    points; ``None`` where a block failed) and a list of failed-block records.
    """
    predicted: list[float | None] = [None] * detection_range
    failed_blocks: list[dict[str, Any]] = []

    offset = 0
    while offset < detection_range:
        anchor_abs = scan_start + offset
        horizon = min(stride, detection_range - offset)
        history = input_data[:anchor_abs]

        result = call_timeseries_service(
            service_url=service_url,
            input_data=history,
            forecast_window=horizon,
            frequency=frequency,
            input_length=input_length,
            target_column=target_column,
        )

        if result.get("status") in ("error", "http_error"):
            failed_blocks.append(
                {"start": offset, "end": offset + horizon - 1, "error": result.get("error", "Unknown error")}
            )
        else:
            preds = result.get("predictions", [])
            usable = min(len(preds), horizon)
            for k in range(usable):
                try:
                    predicted[offset + k] = float(preds[k])
                except (ValueError, TypeError):
                    predicted[offset + k] = None
            if usable < horizon:
                failed_blocks.append(
                    {
                        "start": offset + usable,
                        "end": offset + horizon - 1,
                        "error": f"Service returned only {len(preds)} of {horizon} predictions",
                    }
                )

        offset += horizon

    return predicted, failed_blocks


def _build_severity_curve(
    scan_actual: list[float],
    scan_predicted: list[float | None],
    historical_values: list[float],
    evaluation_window: int,
    drop_weight: float,
    rise_weight: float,
    smoothing: float,
) -> tuple[list[float], list[str]]:
    """Slide the lower window over the scan range and return per-point severity.

    For each anchor index the trailing window (up to ``evaluation_window`` points
    ending at the anchor) is scored; the score is the anchor's severity. The
    window is built only from contiguous points that have a valid prediction, so
    each point is scored exactly once and failed-forecast gaps are skipped.
    """
    n = len(scan_actual)
    severity = [0.0] * n
    directions = ["none"] * n

    for i in range(n):
        if scan_predicted[i] is None:
            continue

        lo = max(0, i - evaluation_window + 1)
        window_actual: list[float] = []
        window_predicted: list[float] = []
        for j in range(lo, i + 1):
            pred_j = scan_predicted[j]
            if pred_j is None:
                # Reset so the window stays contiguous and ends at the anchor.
                window_actual = []
                window_predicted = []
                continue
            window_actual.append(scan_actual[j])
            window_predicted.append(pred_j)

        if not window_actual:
            continue

        score, _ = _evaluate_window(
            actual_values=window_actual,
            predicted_values=window_predicted,
            historical_values=historical_values,
            drop_weight=drop_weight,
            rise_weight=rise_weight,
            smoothing=smoothing,
        )
        severity[i] = score
        directions[i] = "drop" if scan_actual[i] < scan_predicted[i] else "rise"

    return severity, directions


def _segment_intervals(severity: list[float], t_high: float, t_low: float) -> list[tuple[int, int]]:
    """Hysteresis segmentation of a severity curve into [start, end] runs.

    An interval opens when severity rises to/above ``t_high`` and closes only
    when it falls below ``t_low`` (``t_low < t_high``), which suppresses
    chattering around a single threshold.
    """
    intervals: list[tuple[int, int]] = []
    start: int | None = None

    for i, s in enumerate(severity):
        if start is None:
            if s >= t_high:
                start = i
        elif s < t_low:
            intervals.append((start, i - 1))
            start = None

    if start is not None:
        intervals.append((start, len(severity) - 1))

    return intervals


def _merge_intervals(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Merge intervals separated by <= ``gap`` points."""
    if not intervals:
        return []
    merged = [intervals[0]]
    for cur_start, cur_end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start - prev_end - 1 <= gap:
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append((cur_start, cur_end))
    return merged


def detect_anomaly_range(
    input_data: list[float | int | dict[str, Any]],
    detection_range: int,
    frequency: str = "hourly",
    target_column: str = "value",
    input_length: int | None = None,
    stride: int | None = None,
    evaluation_window: int | None = None,
    drop_weight: float = DEFAULT_DROP_WEIGHT,
    rise_weight: float = DEFAULT_RISE_WEIGHT,
    smoothing: float = DEFAULT_SMOOTHING,
    t_high: float = DEFAULT_T_HIGH,
    t_low: float = DEFAULT_T_LOW,
    merge_gap: int = DEFAULT_MERGE_GAP,
    area_min: float = DEFAULT_AREA_MIN,
    peak_high: float = DEFAULT_PEAK_HIGH,
) -> tuple[float, dict[str, Any]]:
    """Scan a long range for anomalies and summarise it into anomalous intervals.

    The trailing ``detection_range`` points are scanned; everything before them is
    used as forecast/context history. The range is forecast in ``stride`` blocks
    (block forecasting), the lower window slides over it to produce a per-point
    severity curve, and the curve is segmented (hysteresis) and filtered (area /
    peak) into zero or more anomalous intervals.

    Args:
        input_data: Continuous (gap-free) time series; the scan range is at the end.
        detection_range: Number of trailing points to scan (must be < len(input_data)).
        frequency: Time series frequency; drives the ``evaluation_window`` and ``stride`` defaults.
        target_column: Column to read from structured (dict) data.
        input_length: Target historical input length for the forecast service per block.
        stride: Block-forecast horizon; defaults from ``frequency`` when None.
        evaluation_window: Lower-window size; defaults from ``frequency`` when None.
        drop_weight: Severity multiplier applied to downward deviations.
        rise_weight: Severity multiplier applied to upward deviations.
        smoothing: Lower-window smoothing strength in [0, 1].
        t_high: Hysteresis upper threshold to open an interval.
        t_low: Hysteresis lower threshold to close an interval.
        merge_gap: Merge intervals separated by <= this many points.
        area_min: Minimum interval area (sum of severity) to keep it.
        peak_high: Peak severity that keeps an interval regardless of its area.

    Returns:
        Tuple of (overall score in [0, 1] = max interval peak, details dict). On
        any failure the score is 0.0 and details contains an ``"error"`` key.
    """
    if evaluation_window is None:
        evaluation_window = _default_evaluation_window(frequency)
    if stride is None:
        stride = _default_stride(frequency)

    if detection_range <= 0:
        return 0.0, {"error": f"detection_range ({detection_range}) must be positive."}
    if not input_data or len(input_data) <= detection_range:
        return 0.0, {
            "error": (
                f"Input data length ({len(input_data)}) must be greater than "
                f"detection_range ({detection_range}) to provide forecast history."
            )
        }

    if not check_forecast_service_health():
        return 0.0, {"error": "Time Series Forecasting Service Unavailable. Cannot perform anomaly detection."}

    service_url = config.get().timeseries_forecasting_service_url

    scan_start = len(input_data) - detection_range
    hist_values = _extract_values(input_data[:scan_start], target_column)
    scan_values = _extract_values(input_data[scan_start:], target_column)

    if not hist_values or len(scan_values) != detection_range:
        return 0.0, {"error": "Could not extract a continuous numeric series for the detection range."}

    predicted, failed_blocks = _block_forecast(
        input_data=input_data,
        scan_start=scan_start,
        detection_range=detection_range,
        stride=stride,
        frequency=frequency,
        input_length=input_length,
        target_column=target_column,
        service_url=service_url,
    )

    forecast_calls = math.ceil(detection_range / stride)

    if all(p is None for p in predicted):
        return 0.0, {
            "error": "All block forecasts failed during range anomaly detection.",
            "failed_blocks": failed_blocks,
        }

    severity, directions = _build_severity_curve(
        scan_actual=scan_values,
        scan_predicted=predicted,
        historical_values=hist_values,
        evaluation_window=evaluation_window,
        drop_weight=drop_weight,
        rise_weight=rise_weight,
        smoothing=smoothing,
    )

    raw_intervals = _segment_intervals(severity, t_high, t_low)
    merged_intervals = _merge_intervals(raw_intervals, merge_gap)

    intervals: list[dict[str, Any]] = []
    for seg_start, seg_end in merged_intervals:
        seg_severity = severity[seg_start : seg_end + 1]
        area = sum(seg_severity)
        peak = max(seg_severity)
        if area < area_min and peak < peak_high:
            continue

        # Dominant direction is the one accumulating the most severity.
        drop_mass = sum(severity[i] for i in range(seg_start, seg_end + 1) if directions[i] == "drop")
        rise_mass = sum(severity[i] for i in range(seg_start, seg_end + 1) if directions[i] == "rise")
        direction = "drop" if drop_mass >= rise_mass else "rise"

        start_abs = scan_start + seg_start
        end_abs = scan_start + seg_end
        peak_offset = seg_start + seg_severity.index(peak)

        intervals.append(
            {
                "start_index": seg_start,
                "end_index": seg_end,
                "start_abs_index": start_abs,
                "end_abs_index": end_abs,
                "start_timestamp": _extract_timestamp(input_data[start_abs]),
                "end_timestamp": _extract_timestamp(input_data[end_abs]),
                "length": seg_end - seg_start + 1,
                "peak": peak,
                "peak_index": peak_offset,
                "peak_timestamp": _extract_timestamp(input_data[scan_start + peak_offset]),
                "mean": area / len(seg_severity),
                "area": area,
                "direction": direction,
                "points": [
                    {
                        "index": i,
                        "actual": scan_values[i],
                        "predicted": predicted[i],
                        "severity": severity[i],
                        "direction": directions[i],
                    }
                    for i in range(seg_start, seg_end + 1)
                ],
            }
        )

    overall_score = max((iv["peak"] for iv in intervals), default=0.0)

    details = {
        "detection_range": detection_range,
        "frequency": frequency,
        "stride": stride,
        "evaluation_window": evaluation_window,
        "forecast_calls": forecast_calls,
        "scan_start_index": scan_start,
        "severity_curve": severity,
        "intervals": intervals,
        "failed_blocks": failed_blocks,
    }

    return overall_score, details


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


def _severity_label(score: float) -> str:
    """Map an anomaly score to a coarse severity label."""
    if score > 0.8:
        return "Critical"
    if score > 0.6:
        return "High"
    if score > 0.4:
        return "Medium"
    return "Low"


def format_anomaly_range_report(score: float, details: dict[str, Any]) -> str:
    """Format a range (sliding) anomaly detection result into a readable report."""
    if "error" in details:
        return f"Anomaly Detection Error: {details['error']}"

    intervals = details.get("intervals", [])
    failed_blocks = details.get("failed_blocks", [])

    report = [
        "🚨 Anomaly Range Detection Report",
        f"Overall Anomaly Score: {score:.4f} (Severity: {_severity_label(score)})",
        "",
        "Scan Summary:",
        f"  • Detection Range: {details.get('detection_range', 0)} points ({details.get('frequency', 'unknown')})",
        f"  • Stride (forecast horizon): {details.get('stride', 0)} | "
        f"Evaluation Window: {details.get('evaluation_window', 0)} | "
        f"Forecast Calls: {details.get('forecast_calls', 0)}",
        f"  • Anomalous Intervals Found: {len(intervals)}",
    ]

    if failed_blocks:
        report.append(
            f"  • ⚠️ Degraded: {len(failed_blocks)} forecast block(s) failed and were skipped "
            "(those points were not scored)."
        )

    report.append("")

    if not intervals:
        report.append("💡 Insight: No significant anomalous interval detected across the scanned range.")
        return "\n".join(report)

    for idx, iv in enumerate(intervals):
        start_label = iv.get("start_timestamp")
        end_label = iv.get("end_timestamp")
        if start_label is not None and end_label is not None:
            span = f"{start_label} → {end_label}"
        else:
            span = f"index {iv['start_index']}..{iv['end_index']}"
        report.extend(
            [
                f"Interval {idx + 1} [{span}] ({iv['direction']}):",
                f"    - Length: {iv['length']} points | Peak: {iv['peak']:.4f} "
                f"(Severity: {_severity_label(iv['peak'])}) | Mean: {iv['mean']:.4f} | Area: {iv['area']:.4f}",
            ]
        )

    report.extend(
        [
            "",
            f"💡 Insight: {len(intervals)} anomalous interval(s) detected. The most severe peaks "
            f"to a score of {score:.4f}.",
        ]
    )

    return "\n".join(report)
