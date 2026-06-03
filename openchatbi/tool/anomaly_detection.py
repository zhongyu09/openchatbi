"""Tool for time series anomaly detection (thin LangChain wrapper)."""

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from openchatbi.analysis.anomaly_detection import (
    _default_evaluation_window,
    detect_anomaly_range,
    evaluate_anomalies,
    format_anomaly_range_report,
    format_anomaly_report,
)
from openchatbi.utils import log

logger = logging.getLogger(__name__)


class AnomalyDetectionInput(BaseModel):
    """Input schema for anomaly detection tool."""

    reasoning: str = Field(description="Reason for using anomaly detection and what insights you expect to gain")
    input_data: list[float | int | dict[str, Any]] = Field(
        description=(
            "Full time series as list of numbers or structured rows with timestamps and values. "
            "The trailing `detection_range` points are the period being analysed; ALL preceding "
            "points are the historical context fed to the forecasting model. "
            "IMPORTANT: provide a CONTINUOUS, gap-free series at the given `frequency`: include one "
            "entry per period from the earliest returned period through the end of the analysis window, "
            "and set periods that had no rows in the query result to 0 (a missing period for a "
            "count/volume metric means 0, and a drop to 0 is exactly the anomaly to catch). Do this "
            "yourself when constructing this argument; do not omit empty periods. If the history is "
            "shorter than the model's minimum input length, the forecasting service backfills the "
            "earliest points automatically."
        )
    )
    detection_range: int | None = Field(
        default=None,
        description=(
            "Number of trailing points to scan for anomalies (the analysis range, e.g. 24 for the "
            "last 24 hourly points). A short internal window slides across this range to locate one "
            "or more anomalous intervals — use this whenever you want to analyse a span of time "
            "rather than just the latest moment. Leave unset to only check the most recent moment "
            "(single-window mode). Must be smaller than the total length of `input_data`."
        ),
        ge=2,
    )
    frequency: str = Field(default="hourly", description="Time series frequency: hourly, daily, weekly, monthly, etc.")
    stride: int | None = Field(
        default=None,
        description=(
            "Block-forecast horizon used while scanning a `detection_range`: the range is forecast in "
            "non-overlapping blocks of this size (about ceil(range/stride) forecast calls). Leave "
            "unset to derive a sensible default from `frequency`."
        ),
        ge=1,
        le=200,
    )
    target_column: str = Field(
        default="value", description="Column name to evaluate for structured data (default: 'value')"
    )
    input_length: int | None = Field(
        default=None,
        description=(
            "Target length of historical input for the forecasting service. If the supplied history "
            "is longer, only the most recent `input_length` points are used; if it is shorter, the "
            "service left-pads the earliest points with zeros to reach this length. Usually leave "
            "unset; the tool sets it automatically when history is below the model minimum."
        ),
    )
    drop_weight: float = Field(
        default=1.0, description="Severity multiplier for downward deviations (>1 emphasises drops)", ge=0.0, le=3.0
    )
    rise_weight: float = Field(
        default=1.0, description="Severity multiplier for upward deviations (>1 emphasises spikes)", ge=0.0, le=3.0
    )


@tool("anomaly_detection", args_schema=AnomalyDetectionInput, return_direct=False, infer_schema=True)
def anomaly_detection(
    reasoning: str,
    input_data: list[float | int | dict[str, Any]],
    detection_range: int | None = None,
    frequency: str = "hourly",
    stride: int | None = None,
    target_column: str = "value",
    input_length: int | None = None,
    drop_weight: float = 1.0,
    rise_weight: float = 1.0,
) -> str:
    """Evaluate time series data for anomalies using forecasting and multi-factor scoring.

    Two modes:
    - Range scan (recommended for analysing a span of time): set ``detection_range`` to the number
      of trailing points to analyse (e.g. 24 hourly points). A short internal window slides across
      the range to produce a per-point anomaly score, which is summarised into one or more anomalous
      intervals (each with its time span, peak/mean score and direction). The range is forecast in
      blocks so far-horizon predictions are not smoothed out.
    - Single moment (default): leave ``detection_range`` unset to only check the most recent moment.

    The score (0 to 1) combines several orthogonal factors: statistical deviation significance,
    deviation direction (drop vs rise), business volume of the moment, the metric's historical
    noisiness, and anomaly duration.

    Provide ``input_data`` as a continuous, gap-free series (missing periods filled with 0 up to the
    end of the analysis window). Leading history before the earliest point is backfilled by the
    forecasting service.

    Args:
        reasoning: Explanation of why anomaly detection is needed
        input_data: Continuous (gap-free) time series; the analysed points are at the end
        detection_range: Number of trailing points to scan; unset = single most-recent moment
        frequency: Time series frequency - hourly, daily, weekly, monthly, etc.
        stride: Block-forecast horizon for range scans; unset = derived from frequency
        target_column: Column name to evaluate for structured data (default: 'value')
        input_length: Target historical input length; shorter history is left-padded with zeros by the service
        drop_weight: Severity multiplier for downward deviations (>1 emphasises drops)
        rise_weight: Severity multiplier for upward deviations (>1 emphasises spikes)

    Returns:
        str: Formatted anomaly detection report with scores and insights
    """
    log(f"Anomaly Detection: {reasoning}")

    if detection_range is not None:
        score, details = detect_anomaly_range(
            input_data=input_data,
            detection_range=detection_range,
            frequency=frequency,
            target_column=target_column,
            input_length=input_length,
            stride=stride,
            drop_weight=drop_weight,
            rise_weight=rise_weight,
        )
        return format_anomaly_range_report(score, details)

    score, details = evaluate_anomalies(
        input_data=input_data,
        evaluation_window=_default_evaluation_window(frequency),
        frequency=frequency,
        target_column=target_column,
        input_length=input_length,
        drop_weight=drop_weight,
        rise_weight=rise_weight,
    )

    return format_anomaly_report(score, details)
