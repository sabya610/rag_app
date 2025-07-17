# RAG QA App with LLaMA, Flask, and pgvector

This is a Retrieval-Augmented Generation (RAG) application that uses `llama.cpp`, `pgvector`, and a Flask API. It supports chat UI, PDF ingestion, and semantic search using PostgreSQL + pgvector.

## 📦 Project Structure

- `app/` — Flask backend code
- `k8s/` — Raw Kubernetes YAMLs
- `helm/rag-qa-chart/` — Production-ready Helm chart
- `Dockerfile` — Containerize the app
- `docker-compose.yml` — Local development stack

## 🚀 Deployment Options

### 1. Docker Compose (Local)
```bash
docker-compose up --build
