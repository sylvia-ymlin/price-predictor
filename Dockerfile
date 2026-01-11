# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Build argument for versioning (also acts as cache buster for training)
ARG MODEL_VERSION=1.0.0
ENV MODEL_VERSION=${MODEL_VERSION}

# Train model during build to ensure compatibility and standalone capability
RUN python scripts/train_and_export.py

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Define environment variable
ENV PYTHONPATH=/app
ENV PORT=8080

# Cloud Run requires listening on PORT env variable
CMD exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT}
