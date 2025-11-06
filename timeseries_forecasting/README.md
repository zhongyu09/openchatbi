# Transformer Time Series Forecasting Service

A Docker-based time series forecasting service using Transformer based models for accurate time series prediction. This service provides a FastAPI-based REST API for easy integration with various applications.

## Features

- **Transformer Model Integration**: Uses state-of-the-art Transformer models for time series forecasting
- **FastAPI Backend**: Modern, fast web framework with automatic API documentation
- **Docker Support**: Fully containerized service for easy deployment
- **Flexible Input**: Supports both simple numeric arrays and structured data with timestamps
- **Multiple Forecast Horizons**: Configure prediction length from 1 to 200 time steps
- **GPU Support**: Automatic GPU detection and utilization when available

## Prerequisites

- Docker installed and running
- Transformer model files (compatible with Hugging Face transformers library)

## Quick Start

### 1. Download Transformer Model

Download a pre-trained model from Hugging Face and place it in the `hf_model` directory. For example, to use the recommended `timer-base-84m` model from https://huggingface.co/thuml/timer-base-84m:

> **Note**: The `timer-base-84m` model requires at least 96 time points in the input data. When integrating with OpenChatBI, add this restriction to your `extra_tool_use_rule` in bi.yaml:
> ```
> - timeseries_forecast tool requires at least 96 time points in input data. Ensure input meets this requirement, or set input_len to 96+ to pad with zeros.
> ```

```bash


### 2. Build and Run

```bash
cd timeseries_forecasting
chmod +x build_and_run.sh
./build_and_run.sh
```

The service will be available at:
- **Predictions**: `http://localhost:8765/predict`
- **Health Check**: `http://localhost:8765/health`
- **API Documentation**: `http://localhost:8765/docs`
- **Model Info**: `http://localhost:8765/model/info`

### 2. Make a Prediction

```bash
curl -X POST http://localhost:8765/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input": [100, 102, 98, 105, 107, 103, 99, 101, 104, 106],
    "input_len": 100,
    "forecast_window": 5,
    "frequency": "H"
  }'
```

### 4. Test the Service

Run the comprehensive test suite:

```bash
python test_forecasting.py --url http://localhost:8765
```

## API Reference

### Prediction Endpoint

**POST** `/predict`

#### Request Format

```json
{
  "input": [...],              // Time series data (required)
  "forecast_window": 24,       // Number of future points to predict (default: 24, max: 200)
  "frequency": "H",           // Frequency: "H" (hourly), "D" (daily), etc. (default: "H")
  "input_len": null,          // Limit input length, if provided, will use it to truncate input or pad zero (optional)
  "target_column": "value"    // Column name for structured data (default: "value")
}
```

#### Input Data Formats

**Simple Numeric Array:**
```json
{
  "input": [100, 102, 98, 105, 107, 103, 99, 101], 
  "input_len": 100,
  "forecast_window": 12
}
```

**Structured Data with Timestamps:**
```json
{
  "input": [
    {"timestamp": "2024-01-01T00:00:00", "value": 100},
    {"timestamp": "2024-01-01T01:00:00", "value": 102},
    {"timestamp": "2024-01-01T02:00:00", "value": 98}
  ],
  "input_len": 100,
  "forecast_window": 24,
  "target_column": "value"
}
```

#### Response Format

```json
{
  "predictions": [101.5, 103.2, 99.8, ...],
  "forecast_window": 24,
  "frequency": "H",
  "status": "success"
}
```

## Configuration

### Environment Variables

- `PYTHONPATH`: Python path for modules (default: /home/model-server)
- `PYTHONUNBUFFERED`: Disable Python output buffering (default: 1)

### Docker Run Options

```bash
# With volume mount for models
docker run -p 8765:8765 \
  -v /path/to/model:/app/hf_model \
  timeseries-forecasting

# With custom environment variables
docker run -p 8765:8765 \
  -e PYTHONPATH=/home/model-server \
  timeseries-forecasting
```

## Testing

### Service Tests

Run the test script to validate the service:

```bash
# Make test script executable
chmod +x test_forecasting.py

# Install test dependencies
pip install requests numpy

# Run tests
python test_forecasting.py --url http://localhost:8765
```

## Model Information

- **Recommended Models**: https://huggingface.co/thuml/timer-base-84m
- **Model Type**: Transformer-based Causal Language Model for Time Series
- **Framework**: Hugging Face Transformers
- **Architecture**: AutoModelForCausalLM
- **Device Support**: Automatic GPU/CPU detection
- **Capabilities**: Univariate time series forecasting with automatic normalization

## Troubleshooting

### Common Issues

1. **Service Not Starting**
   - Check if port 8765 is available: `lsof -i :8765`
   - Verify Docker has sufficient memory allocated (minimum 4GB recommended)
   - Check logs: `docker logs time-series-forecasting-service`

2. **Model Loading Errors**
   - Ensure model files are properly copied during build
   - Check available disk space (models can be several GB)
   - Verify Hugging Face transformers library compatibility

3. **Prediction Errors**
   - Validate input data format
   - Check forecast horizon is reasonable
   - Ensure input data has sufficient length

### Debug Mode

Enable debug logging:

```bash
docker run -p 8765:8765 \
  -e PYTHONPATH=/home/model-server \
  -e LOGGING_LEVEL=DEBUG \
  timeseries-forecasting
```

## Performance

- **Cold Start**: ~10 seconds (model loading)
- **Inference Time**: ~100-300ms per request (varies by input size and model)
- **Memory Usage**: ~2-4GB (depending on input size and model)
- **Concurrent Requests**: Supported (configure workers)

## Limitations

- Maximum forecast window: 200 time points
- Univariate forecasting (single time series)
- Requires minimum input data for reliable predictions, timer-base-84m needs at least 96 time points
- Model-specific context length limitations may apply
