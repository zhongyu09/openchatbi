"""Core algorithms for time series anomaly detection."""

import math
from typing import Any, Dict, List, Tuple, Union


def _extract_values(data: List[Union[float, int, Dict[str, Any]]], target_column: str = "value") -> List[float]:
    """Extract numeric values from input data."""
    values = []
    for item in data:
        if isinstance(item, (int, float)):
            values.append(float(item))
        elif isinstance(item, dict) and target_column in item:
            try:
                values.append(float(item[target_column]))
            except (ValueError, TypeError):
                continue
    return values


def _calculate_reconstruction_error(actual: float, predicted: float) -> float:
    """Calculate normalized reconstruction error between actual and predicted values."""
    # Avoid division by zero
    denominator = max(abs(predicted), 1e-5)
    error = abs(actual - predicted) / denominator
    # Cap the error at 1.0 for scoring purposes
    return min(error, 1.0)


def _calculate_relative_size(actual: float, historical_values: List[float]) -> float:
    """Calculate relative size of the actual value compared to historical average."""
    if not historical_values:
        return 0.5  # Default middle value if no history
        
    hist_mean = sum(historical_values) / len(historical_values)
    denominator = max(abs(hist_mean), 1e-5)
    
    # How many times larger/smaller is it compared to the mean?
    ratio = actual / denominator
    
    # Normalize to 0-1 range using a sigmoid-like function or simple capping
    # If ratio is around 1 (normal), score is 0.5. If ratio is very large, score approaches 1.0
    # If ratio is very small, score approaches 0.0
    try:
        # Prevent OverflowError for very large ratios
        exponent = max(min(ratio - 1.0, 500.0), -500.0)
        normalized_size = 1.0 - (1.0 / (1.0 + math.exp(exponent)))
    except OverflowError:
        normalized_size = 1.0 if ratio > 1.0 else 0.0
        
    return normalized_size


def _calculate_bound_violation(actual: float, predicted: float, historical_values: List[float]) -> float:
    """Calculate bound violation score. 1.0 if outside bounds, 0.0 if inside."""
    if not historical_values or len(historical_values) < 2:
        # Fallback to simple percentage bound (e.g., 20% deviation)
        bound_margin = abs(predicted) * 0.2
    else:
        # Calculate standard deviation
        mean = sum(historical_values) / len(historical_values)
        variance = sum((x - mean) ** 2 for x in historical_values) / (len(historical_values) - 1)
        std_dev = math.sqrt(variance)
        # Use 3 sigma as bound
        bound_margin = 3 * std_dev
        
    lower_bound = predicted - bound_margin
    upper_bound = predicted + bound_margin
    
    if actual < lower_bound or actual > upper_bound:
        # Calculate how far outside the bound it is (normalized)
        deviation = min(abs(actual - predicted) / max(bound_margin, 1e-5), 2.0)
        # Map deviation [1.0, 2.0] to score [0.5, 1.0]
        return 0.5 + (deviation - 1.0) * 0.5
    
    return 0.0


def _evaluate_window(
    actual_values: List[float], 
    predicted_values: List[float], 
    historical_values: List[float]
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a window of recent data points to calculate an overall anomaly score.
    Newer data points have higher weights.
    """
    if not actual_values or not predicted_values:
        return 0.0, {"error": "Empty values provided for evaluation"}
        
    window_size = min(len(actual_values), len(predicted_values))
    
    # Weights for the window (newer data gets higher weight)
    # e.g., for window=3: weights could be [0.2, 0.3, 0.5]
    weights = [math.exp(i) for i in range(window_size)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    overall_score = 0.0
    point_details = []
    consecutive_anomalies = 0
    
    for i in range(window_size):
        actual = actual_values[i]
        predicted = predicted_values[i]
        weight = normalized_weights[i]
        
        # Calculate individual factors
        recon_error = _calculate_reconstruction_error(actual, predicted)
        rel_size = _calculate_relative_size(actual, historical_values)
        bound_violation = _calculate_bound_violation(actual, predicted, historical_values)
        
        # Combine factors for this specific point
        # Weights for factors: Reconstruction Error (40%), Bound Violation (40%), Relative Size (20%)
        point_score = (recon_error * 0.4) + (bound_violation * 0.4) + (rel_size * 0.2)
        
        # Track consecutive anomalies (threshold > 0.6)
        if point_score > 0.6:
            consecutive_anomalies += 1
        else:
            consecutive_anomalies = 0
            
        overall_score += point_score * weight
        
        point_details.append({
            "actual": actual,
            "predicted": predicted,
            "reconstruction_error": recon_error,
            "bound_violation": bound_violation,
            "relative_size": rel_size,
            "point_score": point_score,
            "weight": weight
        })
        
    # Duration factor: boost score if multiple consecutive anomalies
    duration_boost = 0.0
    if consecutive_anomalies > 1:
        duration_boost = min((consecutive_anomalies - 1) * 0.1, 0.2)
        
    final_score = min(overall_score + duration_boost, 1.0)
    
    details = {
        "window_size": window_size,
        "consecutive_anomalies": consecutive_anomalies,
        "duration_boost": duration_boost,
        "point_details": point_details
    }
    
    return final_score, details


from openchatbi.tool.timeseries_forecast import call_timeseries_service, _check_service_health

def evaluate_anomalies(
    input_data: List[Union[float, int, Dict[str, Any]]],
    evaluation_window: int = 3,
    frequency: str = "hourly",
    target_column: str = "value",
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate time series data for anomalies using forecasting and multi-factor scoring."""
    # Get service URL from config
    from openchatbi import config
    service_url = config.get().timeseries_forecasting_service_url
    
    if not input_data or len(input_data) <= evaluation_window:
        return 0.0, {"error": f"Input data length ({len(input_data)}) must be greater than evaluation window ({evaluation_window})."}
        
    # Check service availability
    if not _check_service_health(service_url):
        return 0.0, {"error": "Time Series Forecasting Service Unavailable. Cannot perform anomaly detection without the forecasting service."}
        
    # Split data into historical (training) and evaluation window
    historical_data = input_data[:-evaluation_window]
    evaluation_data = input_data[-evaluation_window:]
    
    # Extract numeric values
    hist_values = _extract_values(historical_data, target_column)
    eval_values = _extract_values(evaluation_data, target_column)
    
    if not hist_values or not eval_values:
        return 0.0, {"error": "Could not extract numeric values from input data."}
        
    # Call forecasting service to predict the evaluation window
    forecast_result = call_timeseries_service(
        service_url=service_url,
        input_data=historical_data,
        forecast_window=evaluation_window,
        frequency=frequency,
        target_column=target_column
    )
    
    if forecast_result.get("status") in ("error", "http_error"):
        return 0.0, {"error": f"Forecasting Error during anomaly detection: {forecast_result.get('error', 'Unknown error')}"}
        
    predicted_values = forecast_result.get("predictions", [])
    
    if not predicted_values or len(predicted_values) != len(eval_values):
        return 0.0, {"error": f"Mismatch between predicted values ({len(predicted_values)}) and evaluation window size ({len(eval_values)})."}
        
    # Evaluate the window
    return _evaluate_window(
        actual_values=eval_values,
        predicted_values=predicted_values,
        historical_values=hist_values
    )

def format_anomaly_report(score: float, details: Dict[str, Any], reasoning: str) -> str:
    """Format the anomaly detection result into a readable report."""
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
        f"  • Consecutive Anomalies: {details.get('consecutive_anomalies', 0)}",
        f"  • Duration Boost Applied: +{details.get('duration_boost', 0.0):.4f}",
        "",
        "Detailed Point Evaluation (Oldest to Newest):"
    ]
    
    for i, point in enumerate(details.get("point_details", [])):
        report.extend([
            f"  Point {i + 1} (Weight: {point['weight']:.2f}):",
            f"    - Actual: {point['actual']:.2f} | Predicted: {point['predicted']:.2f}",
            f"    - Reconstruction Error: {point['reconstruction_error']:.4f}",
            f"    - Bound Violation Score: {point['bound_violation']:.4f}",
            f"    - Relative Size Score: {point['relative_size']:.4f}",
            f"    - Point Anomaly Score: {point['point_score']:.4f}"
        ])
        
    if score > 0.6:
        report.extend([
            "",
            "💡 Insight: Significant anomaly detected. The actual values deviate substantially from the expected pattern."
        ])
    else:
        report.extend([
            "",
            "💡 Insight: No significant anomaly detected. The data follows the expected pattern closely."
        ])
        
    return "\n".join(report)
