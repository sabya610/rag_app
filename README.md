# RAG QA App with LLaMA, Flask, and pgvector

A Retrieval-Augmented Generation (RAG) application that uses `llama.cpp`, `pgvector`, and a Flask API. Supports conversational chat, PDF ingestion, and semantic search over PostgreSQL + pgvector with offline embeddings.

**Latest Version**: v10 (llama3.1-8b-q4) — Production deployed in Kubernetes

## Recent Improvements (v10)

Enhanced RAG generation to preserve detailed information from Knowledge Articles:

1. **Context Window Expansion** — LLM context: 8192 → 16384 tokens (doubled capacity)
2. **Chunk Retrieval Increase** — MAX_RESULTS: 5 → 15 (retrieves 3x more context)
3. **Context Size Increase** — MAX_CONTEXT_CHARS: 12000 → 20000 (67% more input)
4. **Adjacent Chunk Retrieval** — Preserves document context via neighboring chunks (±1)
5. **Enhanced System Prompt** — 7-rule prompt explicitly prevents over-summarization
6. **Temperature Reduction** — 0.2 → 0.1 (more deterministic, less hallucination)

**Result**: Less over-summarization, more detailed troubleshooting steps, better faithfulness to source material.

## Project Structure

```
rag_app/
├── app/
│   ├── __init__.py          # Flask app initialization, LLM model loading
│   ├── config.py            # Configuration & limits
│   ├── utils.py             # Core RAG utilities (retrieval, inference, processing)
│   ├── models.py            # Database models
│   ├── routes/
│   │   └── rag_routes.py    # Flask routes for Q&A endpoints
│   ├── services/
│   │   ├── sfdc_client.py   # Salesforce integration
│   │   └── sfdc_knowledge.py # SFDC Knowledge Article retrieval
│   ├── templates/
│   │   ├── index.html       # Chat UI
│   │   └── history.html     # Conversation history
│   └── models/
│       └── embedding/all-MiniLM-L6-v2/  # Offline embeddings (384-dim, 80MB)
├── helm/                    # Kubernetes Helm charts (production-ready)
├── postgres/                # PostgreSQL deployment manifests
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image
└── run.py                   # Flask entry point
```

## Stack

- **LLM**: Meta-Llama-3.1-8B-Instruct-Q4_K_M (4-bit quantized, 7.5GB)
- **Inference**: llama.cpp (C++ backend with n_ctx parameter)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (offline, 384-dim)
- **Vector DB**: PostgreSQL + pgvector extension
- **API Framework**: Flask (Python 3.10)
- **Deployment**: Kubernetes + Helm

## Deployment

### Kubernetes (Production)

```bash
# Deploy with Helm
helm install rag-app helm/rag-app -n ragapp

# Or update image version
kubectl set image deployment/rag-app-rag -n ragapp rag=sabya610/rag-app:llama3.1-8b-q4-v10

# Verify deployment
kubectl get pods -n ragapp
kubectl logs -n ragapp -f deployment/rag-app-rag
```

### Docker (Local/Development)

```bash
# Build image
docker build -t sabya610/rag-app:latest .

# Push to registry
docker push sabya610/rag-app:latest

# Run container
docker run -p 5000:5000 \
  -e FLASK_ENV=development \
  -e POSTGRES_URL="postgresql://user:pass@postgres:5432/rag_db" \
  sabya610/rag-app:latest
```

### Docker Images Available

- `sabya610/rag-app:llama3.1-8b-q4-v10` — Latest (v10 improvements)
- `sabya610/rag-app:llama3.1-8b-q4-v9` — Previous stable
- Earlier versions: v8, v7, v6, v5, v4, v3, v2

## Configuration

Key environment variables in `rag_app/app/config.py`:

```python
MAX_RESULTS = 15              # Number of chunks to retrieve (was 5)
MAX_CONTEXT_CHARS = 20000     # Context size in chars (was 12000)
TEMPERATURE = 0.1             # LLM temperature (was 0.2)
N_CTX = 16384                 # LLM context window (was 8192)
```

Set via `.env` file or Kubernetes ConfigMap:
```bash
export MAX_RESULTS=15
export MAX_CONTEXT_CHARS=20000
export TEMPERATURE=0.1
export POSTGRES_URL=postgresql://...
export SFDC_SESSION_ID=...    # Salesforce session token
```

## Integration with Salesforce Knowledge Base

The RAG app integrates with Salesforce Knowledge Articles (Product Line: HPE Ezmeral):

- **Product Mapping**: ezmeral→"CONT PLT SW (RM)", datafabric→"DATA FABRIC SW (PU)"
- **Query Filter**: Articles automatically filtered by `GSD_KM_Issue_Solution_Product_Queue__c = 'HPE Ezmeral'`
- **API Version**: v59.0 on https://hp.my.salesforce.com

Authentication via `SFDC_SESSION_ID` environment variable (Bearer token).

## Database Setup

PostgreSQL + pgvector extension required. Create database:

```sql
CREATE DATABASE rag_db;
\c rag_db
CREATE EXTENSION pgvector;

-- Tables created automatically on app startup
-- kb_chunks: Vector embeddings of knowledge base chunks
-- conversation_history: Chat history storage
```

### Reset Embedded Data

If chunks are embedded incorrectly, reset the table:

```sql
DELETE FROM kb_chunks;
-- App will re-embed on next ingest
```

## API Endpoints

### Q&A Endpoint
```
POST /api/rag/qa
Content-Type: application/json

{
  "question": "How do I install HPE Ezmeral?"
}

Response:
{
  "answer": "To install HPE Ezmeral...",
  "sources": ["Article 000123", "Article 000456"],
  "model_info": "llama3.1-8b-q4-v10"
}
```

### Chat History
```
GET /api/rag/history
```

### Health Check
```
GET /health
```

## Development

### Local Setup

```bash
# Clone repo
git clone https://github.com/sabya610/rag_app.git
cd rag_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POSTGRES_URL="postgresql://user:pass@localhost:5432/rag_db"
export SFDC_SESSION_ID="your_sfdc_token"

# Run Flask app
python run.py
```

Visit http://localhost:5000 in browser.

### Testing Improvements

Compare v10 output vs. previous versions:
1. Query a troubleshooting article
2. Verify detailed steps are preserved (not over-summarized)
3. Check that command blocks are shown exactly as documented
4. Confirm multiple diagnostic commands included in response

## Performance Notes

- **Embeddings**: Offline (no API calls), ~50ms per query
- **LLM Inference**: ~2-5 seconds per response (8B model, 4-bit quantized)
- **Vector Search**: <10ms for pgvector queries
- **Total Latency**: ~3-6 seconds per Q&A request

## Security

- Session ID stored in Kubernetes secrets
- PostgreSQL credentials in encrypted ConfigMaps
- Model files served from container (no external model downloads)
- No private data sent to external APIs (all embeddings/inference local)

## License

Internal HPE project

## Contributing

Push to `feature/meeting-transcript-rag` branch, then create pull request.

Current branch: `feature/meeting-transcript-rag` with v10 improvements
Latest commit: 14a0b55 (Push v10 improvements...)

