# Dockerfile for Vertex AI Training
# This container will be used to train model_balanced on Vertex AI

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY vertex_ai/requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/
COPY setup.py .

# Install package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:${PYTHONPATH}

# Entry point
ENTRYPOINT ["python", "-m", "src.training.train"]

# Default arguments (can be overridden by Vertex AI)
CMD ["--config", "config/training_config.yaml"]