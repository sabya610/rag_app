# Base Python image
FROM python:3.10-slim

# Install system dependencies (needed for llama-cpp-python build)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy and install Python dependencies
COPY rag_meeting_app/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi

# Copy application source code
COPY rag_meeting_app /app/rag_meeting_app

# Copy models (embedding + LLM)
COPY models /app/models

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server via uvicorn
CMD ["uvicorn", "rag_meeting_app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
