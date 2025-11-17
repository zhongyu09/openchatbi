"""app.py: FastAPI application for Transformer time series forecasting."""

import logging
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from model_handler import TransformerModelHandler, get_model_handler
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Transformer Time Series Forecasting API",
    description="A REST API for time series forecasting using Transformer model",
    version="1.0.0",
)


# Request models
class ForecastRequest(BaseModel):
    """Request model for forecasting."""

    input: list[float | int | dict[str, Any]] = Field(
        ...,
        description="Time series data as list of numbers or structured data",
        example=[100, 102, 98, 105, 107, 103, 99, 101],
    )
    forecast_window: int = Field(default=24, ge=1, le=200, description="Number of future points to predict")
    input_len: int | None = Field(default=None, description="Optional input length limit")
    frequency: str = Field(default="hourly", description="Frequency of the time series (hourly, daily, etc.)")
    target_column: str = Field(default="value", description="Column name for structured data")


class ForecastResponse(BaseModel):
    """Response model for forecasting."""

    predictions: list[float] = Field(description="Forecasted values")
    forecast_window: int = Field(description="Number of predictions")
    frequency: str = Field(description="Time series frequency")
    status: str = Field(description="Response status")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(description="Error message")
    status: str = Field(description="Response status")


# Global variables
model_handler: TransformerModelHandler | None = None
startup_time: float | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model_handler, startup_time
    startup_time = time.time()
    logger.info("Starting Transformer Forecasting API...")

    try:
        # Initialize model handler
        model_handler = get_model_handler()
        model_success = model_handler.initialize()

        if model_success:
            logger.info("Model initialized successfully")
        else:
            logger.error("Failed to initialize model")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time if startup_time else 0

    return {
        "status": "healthy",
        "model_initialized": model_handler.initialized if model_handler else False,
        "uptime_seconds": round(uptime, 2),
    }


@app.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=ForecastResponse | ErrorResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
)
async def predict(request: ForecastRequest):
    """
    Main forecasting endpoint.

    Args:
        request: Forecast request containing time series data and parameters

    Returns:
        Forecast response with predictions or error
    """
    try:
        logger.info(f"Received prediction request: {len(request.input)} data points, horizon={request.forecast_window}")

        # Check if model is initialized
        if not model_handler or not model_handler.initialized:
            raise HTTPException(status_code=500, detail="Model not initialized")

        # Validate input
        if len(request.input) == 0:
            raise HTTPException(status_code=400, detail="Input data cannot be empty")

        # Make prediction
        result = model_handler.predict(
            time_series_data=request.input,
            forecast_window=request.forecast_window,
            input_len=request.input_len,
            frequency=request.frequency,
            target_column=request.target_column,
        )

        # Check if prediction was successful
        if result.get("status") == "error":
            raise HTTPException(status_code=result.get("code", 500), detail=result.get("error", "Prediction failed"))

        logger.info(f"Prediction successful: {len(result['predictions'])} predictions generated")

        return ForecastResponse(**result)

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content=ErrorResponse(error=str(e), status="error").model_dump())
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return ErrorResponse(error=str(e), status="error")


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if not model_handler or not model_handler.initialized:
        return {"error": "Model not initialized", "status": "error"}

    return {
        "model_path": model_handler.model_path,
        "device": str(model_handler.device),
        "initialized": model_handler.initialized,
        "config": str(model_handler.config) if model_handler.config else None,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Transformer Time Series Forecasting API",
        "version": "1.0.0",
        "description": "REST API for time series forecasting using Transformer model",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "ping": "/ping",
            "model_info": "/model/info",
            "docs": "/docs",
        },
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {"error": exc.detail, "status": "error", "status_code": exc.status_code}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error", "status": "error", "status_code": 500}


if __name__ == "__main__":
    # For development
    uvicorn.run("app:app", host="0.0.0.0", port=8765, reload=True, log_level="info")
