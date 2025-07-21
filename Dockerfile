# Base Python image
FROM python:3.10-slim

# System Dependencies
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

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and static files
COPY . .

# Set environment variables
ENV PYTHONBUFFERED=1
ENV FLASK_APP=rag_app/rag_qa_ui_pgvector_hist.py
ENV TRANSFORMERS_OFFLINE=1

# Create model directories and download embedding model
RUN mkdir -p /models/embedding && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/models/embedding')"

# Copy LLaMA model
COPY models/llama-2-7b-chat.Q4_K_M.gguf /models/llama-2-7b-chat.Q4_K_M.gguf

# Copy PDF files (optional)
COPY pdf_kb_files /pdf_kb_files

# Expose port
EXPOSE 5000

# Run app using gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "rag_app.rag_qa_ui_pgvector_hist:app"]
