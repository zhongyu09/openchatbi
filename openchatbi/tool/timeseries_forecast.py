"""Tool for time series forecasting."""

import logging
from typing import Any

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.utils import log

logger = logging.getLogger(__name__)


class TimeseriesForecastInput(BaseModel):
    """Input schema for time series forecasting tool."""

    reasoning: str = Field(description="Reason for using time series forecasting and what insights you expect to gain")
    input_data: list[float | int | dict[str, Any]] = Field(
        description="Time series data as list of numbers or structured data with timestamps and values"
    )
    forecast_window: int = Field(
        default=24, description="Number of future time points to predict (1-200)", ge=1, le=200
    )
    frequency: str = Field(default="hourly", description="Time series frequency: hourly, daily, weekly, monthly, etc.")
    input_length: int | None = Field(
        default=None, description="Optional limit on input data length to use for prediction"
    )
    target_column: str = Field(
        default="value", description="Column name to forecast for structured data (default: 'value')"
    )


def _check_service_health(service_url: str) -> bool:
    """Check if time series forecasting service is available."""
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get("model_initialized", False)
        return False
    except requests.exceptions.RequestException:
        return False


def check_forecast_service_health() -> bool:
    try:
        service_url = config.get().timeseries_forecasting_service_url
        return _check_service_health(service_url)
    except ValueError:
        # Configuration not loaded yet (e.g., in tests)
        return False


def _call_timeseries_service(
    service_url: str,
    input_data: list[float | int | dict[str, Any]],
    forecast_window: int,
    frequency: str,
    input_length: int | None = None,
    target_column: str = "value",
) -> dict[str, Any]:
    """Call time series forecasting service."""
    try:
        # Prepare request payload
        payload = {"input": input_data, "forecast_window": forecast_window, "frequency": frequency}

        if input_length is not None:
            payload["input_len"] = input_length

        if target_column != "value":
            payload["target_column"] = target_column

        # Make request to time series forecasting service
        response = requests.post(f"{service_url}/predict", json=payload, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Service returned status {response.status_code}: {response.text}",
                "status": "http_error",
                "status_code": response.status_code,
            }

    except requests.exceptions.Timeout:
        return {"error": "Request timeout - forecasting service took too long to respond", "status": "error"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to forecasting service: {str(e)}", "status": "error"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "status": "error"}


def _format_forecast_result(result: dict[str, Any], reasoning: str, input_data_length: int) -> str:
    """Format the forecasting result for the agent."""
    if result.get("status") == "error":
        return f"""Time Series Forecasting Error: {result.get('error', 'Unknown error occurred')}
Please check:
1. Time series forecasting service is running (docker run -p 8765:8765 timeseries-forecasting)
2. Model load successfully
3. Try again if timeout"""
    elif result.get("status") == "http_error":
        if result.get("status_code") == 400:
            return f"""Time Series Forecasting Error: {result.get('error', 'Unknown error occurred')}
Please check:
1. Input data format is correct
2. input_len is set to larger when the input data length is not enough
3. Forecast window is reasonable (1-200)"""
        else:
            return f"""Time Series Forecasting Error: {result.get('error', 'Unknown error occurred')}"""

    predictions = result.get("predictions", [])
    forecast_window = result.get("forecast_window", len(predictions))
    frequency = result.get("frequency", "unknown")

    if not predictions:
        return "No predictions were generated. Please check your input data."

    # Calculate basic statistics
    sum_predictions = sum(predictions)
    avg_prediction = sum_predictions / len(predictions) if predictions else 0
    min_prediction = min(predictions) if predictions else 0
    max_prediction = max(predictions) if predictions else 0

    # Create formatted response
    response_parts = [
        "✅ Time Series Forecasting Completed",
        "",
        "Forecast Summary:",
        f"  • Input data points: {input_data_length}",
        f"  • Forecast window: {forecast_window} {frequency.lower()} periods",
        "",
        "Predictions:",
        f"  • Average forecast: {avg_prediction:.2f}",
        f"  • Sum: {sum_predictions:.2f}",
        f"  • Range: {min_prediction:.2f} to {max_prediction:.2f}",
        f"  • Total periods forecasted: {len(predictions)}",
        "",
        "Detailed Forecast Values:",
    ]

    for i, pred in enumerate(predictions):
        period_label = f"Period {i + 1}"
        response_parts.append(f"  • {period_label}: {pred:.2f}")

    return "\n".join(response_parts)


@tool("timeseries_forecast", args_schema=TimeseriesForecastInput, return_direct=False, infer_schema=True)
def timeseries_forecast(
    reasoning: str,
    input_data: list[float | int | dict[str, Any]],
    forecast_window: int = 24,
    frequency: str = "hourly",
    input_length: int | None = None,
    target_column: str = "value",
) -> str:
    """Forecast future values for time series data using advanced deep learning models.

    This tool uses state-of-the-art deep learning models (currently transformer based) to predict future values based on historical time series data.
    Perfect for sales forecasting, demand planning, trend analysis, and business intelligence.

    Args:
        reasoning: Explanation of why forecasting is needed and what insights are expected
        input_data: Historical time series data as list of numbers or structured data with timestamps
        forecast_window: Number of future time points to predict (1-200, default: 24)
        frequency: Time series frequency - hourly, daily, weekly, monthly, etc.
        input_length: Optional limit on how much historical data to use for prediction
        target_column: Column name to forecast for structured data (default: 'value')

    Returns:
        str: Formatted forecast results with predictions, statistics, and interpretation guidance

    Examples:
        - Sales forecasting: Predict next month's daily sales based on historical data
        - Demand planning: Forecast product demand for inventory management
        - Financial planning: Predict revenue, costs, or other financial metrics
        - Operational planning: Forecast website traffic, resource usage, etc.
    """

    # Get service URL from config
    service_url = config.get().timeseries_forecasting_service_url

    log(f"Time Series Forecast: {reasoning}")
    log(f"Input data points: {len(input_data)}, Forecast window: {forecast_window}, Frequency: {frequency}")

    # Validate input data
    if not input_data:
        return "Error: Input data cannot be empty. Please provide historical time series data."

    if len(input_data) < 3:
        return "Error: Need at least 3 data points for reliable forecasting. Please provide more historical data."

    # Check service availability
    if not _check_service_health(service_url):
        return """Time Series Forecasting Service Unavailable. The time series forecasting service is not running or not in service. """

    # Call the forecasting service
    result = _call_timeseries_service(
        service_url=service_url,
        input_data=input_data,
        forecast_window=forecast_window,
        frequency=frequency,
        input_length=input_length,
        target_column=target_column,
    )

    # Format and return the result
    return _format_forecast_result(result, reasoning, len(input_data))
