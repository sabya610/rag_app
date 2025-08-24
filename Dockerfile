# Base Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=1000 torch==2.8.0 triton==3.4.0
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy application source code
COPY rag_app ./rag_app
COPY .env ./rag_app
#COPY run.py .
#COPY templates ./templates
#COPY __init__.py .

# Copy additional resources into container
COPY models ./models
COPY postgres ./postgres
COPY pdf_kb_files ./pdf_kb_files
COPY helm ./helm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=run.py \
    TRANSFORMERS_OFFLINE=1

# Expose Flask/Gunicorn port
EXPOSE 5000

# Run app with Gunicorn (using run.py as entrypoint)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "run:app"]
