#!/bin/bash

# Build and run script for time series forecasting service
set -e

echo "=== Building Timeseries Forecasting Docker Container ==="

# Check if the hf_model model directory exists
MODEL_DIR="../hf_model"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Hugging face model directory not found at $MODEL_DIR"
    echo "Please ensure the model is downloaded and available at this location"
    exit 1
fi

echo "✓ Found Hugging face model at: $MODEL_DIR"

rm -rf hf_model
cp -r $MODEL_DIR .

# Build the Docker image
echo "Building Docker image..."
docker build -t timeseries-forecasting .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
else
    echo "✗ Failed to build Docker image"
    exit 1
fi

# Check if container is already running
CONTAINER_NAME="time-series-forecasting-service"
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

echo "=== Starting Timeseries Forecasting Service ==="

# Run the container
docker run -d \
    --name $CONTAINER_NAME \
    -p 8765:8765 \
    timeseries-forecasting

if [ $? -eq 0 ]; then
    echo "✓ Container started successfully"
    echo ""
    echo "Service endpoints:"
    echo "  - Predictions: http://localhost:8765/predict"
    echo "  - Health Check: http://localhost:8765/health"
    echo "  - API Docs: http://localhost:8765/docs"
    echo ""
    echo "Container logs:"
    echo "  docker logs -f $CONTAINER_NAME"
    echo ""
    echo "To test the service:"
    echo "  python test_forecasting.py"
    echo ""
    echo "To stop the service:"
    echo "  docker stop $CONTAINER_NAME"
else
    echo "✗ Failed to start container"
    exit 1
fi

# Wait a moment and check if container is still running
sleep 5
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "✓ Service is running"

    # Show few logs
    echo ""
    echo "=== Initial Service Logs ==="
    docker logs "$CONTAINER_NAME" | head -n 50
else
    echo "✗ Service failed to start"
    echo "Checking logs..."
    docker logs $CONTAINER_NAME
    exit 1
fi