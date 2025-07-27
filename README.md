# RAG QA App with LLaMA, Flask, and pgvector

This is a Retrieval-Augmented Generation (RAG) application that uses `llama.cpp`, `pgvector`, and a Flask API. It supports chat UI, PDF ingestion, and semantic search using PostgreSQL + pgvector.

## 📦 Project Structure

- `rag_app/` — Flask backend code
- `helm/` — Dummy Helm chart for reference
- `postgres/` - Production ready postgres Database yaml file for deployment in kubernetes env Eg kubectl apply -f *.yaml
- `Dockerfile` — Containerize the app
- `rag-app-0.1.0.tgz | rag-app`  — rag-app Helm chart and package created having all the production ready rag-app yaml
- `requirement.txt` - All the packages to be installed
- `rag_app` - All the rag-app tools script and LLM models and Embedding Models repositories
## Postgres db need to installed as prerequisites

## 🚀 Deployment Options of rag-app in kubernetes env or helm package to be deployed in PCAI setup
- `helm package /root/rag-app --destination /root`
- `helm install rag-app rag-app-0.1.0.tgz`

###  Docker images (Local)
- docker build sabya610/rag-app:latest .
- docker push sabya610/rag-app

### Delete and CLean UP kb_chunks tables .Clear and Re-embed Your Database
Because your existing chunks are already embedded incorrectly, you must reset the kb_chunks table.
Run this in your DB (e.g., psql or PgAdmin):

sql
DELETE FROM kb_chunks;

