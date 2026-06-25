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

# Install with trusted hosts to handle corporate proxy SSL interception
RUN pip install --default-timeout=1000 --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy application source code
COPY rag_app ./rag_app
COPY .env ./rag_app

# Copy additional resources into container
COPY models /app/rag_app/models
COPY postgres /app/rag_app/postgres
COPY pdf_kb_files /app/rag_app/pdf_kb_files
COPY helm /app/rag_app/helm
COPY models/embedding/all-MiniLM-L6-v2 /app/rag_app/models/embedding/all-MiniLM-L6-v2

# Create SFDC secrets mount point (K8s Secret will mount sid.txt here)
RUN mkdir -p /etc/sfdc

# Copy sid.txt if available at build time (optional, for non-K8s deploys)
# In K8s, this is overridden by the Secret volume mount
COPY sid.tx[t] /etc/sfdc/

# Set environment variables (platform-agnostic)
ENV PYTHONPATH=/app/rag_app \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=run.py \
    TRANSFORMERS_OFFLINE=1 \
    SF_SID_FILE=/etc/sfdc/sid.txt


# Expose Flask/Gunicorn port
EXPOSE 5000

# Run app with Gunicorn (using run.py as entrypoint)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "300", "--preload","rag_app.run:app"]
