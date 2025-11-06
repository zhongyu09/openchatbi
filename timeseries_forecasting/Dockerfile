FROM python:3.10-slim

# Install only essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies for time series forecasting
RUN pip3 --no-cache-dir install \
    fastapi==0.120.4 \
    uvicorn==0.38.0 \
    transformers==4.40.1 \
    torch==2.9.0 \
    numpy==2.2.6 \
    pandas==2.3.3 \
    pydantic==2.12.3

# Set working directory
WORKDIR /home/model-server

# Copy the model
COPY ../hf_model /home/model-server/hf_model

# Copy application files
COPY app.py model_handler.py /home/model-server/

# Set environment variables
ENV PYTHONPATH=/home/model-server
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8765

# Define entrypoint and default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]