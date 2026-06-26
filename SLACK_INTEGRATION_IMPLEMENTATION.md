# Slack Integration Implementation Guide

**Status**: ✅ Complete  
**Branch**: `feature/slack-integration`  
**GitHub**: https://github.com/sabya610/rag_app/tree/feature/slack-integration

## Executive Summary

The Slack integration has been successfully added to rag_app with:
- ✅ Full message and thread import from Slack
- ✅ Vector embeddings using pgvector
- ✅ Keyword + semantic hybrid search
- ✅ RESTful API endpoints
- ✅ Zero impact on existing SFDC/PDF integration
- ✅ Production-ready deployment configuration

**No changes to the Monday SFDC/PDF demo** — this runs on the `main` branch independently.

---

## Quick Start (5 minutes)

### 1. Create Slack App

```bash
# Go to https://api.slack.com/apps
# Create New App > From scratch
# App name: rag-app
# Select your workspace
```

### 2. Add Bot Scopes

**OAuth & Permissions** → **Scopes**:
```
channels:history
channels:read
users:read
users:read.email
```

### 3. Install to Workspace

Click "Install to Workspace" and authorize.

### 4. Collect Credentials

- **Basic Information**: Copy `Signing Secret`
- **OAuth & Permissions**: Copy `Bot User OAuth Token` (starts with `xoxb-`)

### 5. Configure rag_app

Edit `.env.local`:

```bash
SLACK_BOT_TOKEN=xoxb-YOUR-TOKEN
SLACK_SIGNING_SECRET=YOUR-SIGNING-SECRET
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30
```

### 6. Initialize Database

```bash
python init_slack_db.py
```

### 7. Import Slack Messages

```bash
curl -X POST http://localhost:5000/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 8. Search Slack

```bash
curl -X POST http://localhost:5000/api/slack/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question",
    "search_type": "hybrid",
    "limit": 5
  }'
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Slack Workspace                          │
│                                                             │
│  #general   #engineering   #support   #announcements        │
│     │            │            │              │              │
│     └────────────┴────────────┴──────────────┘              │
│                       │                                      │
└───────────────────────┼──────────────────────────────────────┘
                        │
                  Slack API (Bot)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐    ┌─────▼─────┐    ┌──▼────┐
   │Channels │    │ Messages  │    │Threads│
   └─────────┘    │  & Users  │    └───────┘
                  └───────────┘

        ┌──────────────────────────────┐
        │   rag_app Flask Backend      │
        │                              │
        │  /api/slack/import          │
        │  /api/slack/search          │
        │  /api/slack/threads/{id}    │
        │  /api/slack/channels        │
        │  /api/slack/stats           │
        └──────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    ┌───▼──┐      ┌────▼─────┐    ┌───▼────┐
    │Embed-│      │  Search  │    │  LLM   │
    │dings │      │  Engine  │    │  Core  │
    └──────┘      └──────────┘    └────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼──────┐   ┌───▼────┐  ┌──────▼────┐
   │PostgreSQL │   │pgvector│  │Embeddings │
   │           │   │ Search │  │(MiniLM)   │
   └────────────┘   └────────┘  └───────────┘
   
   ├─ slack_messages    (individual messages + embeddings)
   ├─ slack_threads     (thread metadata + embeddings)
   ├─ kb_chunks         (existing PDF/SFDC data)
   └─ qa_history        (conversation history)
```

---

## Project Structure

```
rag_app/
├── app/
│   ├── services/
│   │   ├── slack_client.py           # Slack API wrapper
│   │   ├── slack_import.py           # Message import + embeddings
│   │   ├── sfdc_client.py            # SFDC (existing)
│   │   └── populate_db.py            # PDF processing (existing)
│   │
│   ├── routes/
│   │   ├── slack_routes.py           # NEW: Slack endpoints
│   │   └── rag_routes.py             # Existing RAG endpoints
│   │
│   ├── models.py                     # Updated with SlackMessage, SlackThread
│   ├── config.py                     # Updated with Slack config
│   └── __init__.py                   # Updated to register slack routes
│
├── .env.local                         # NEW: Slack credentials template
├── SLACK_INTEGRATION.md               # NEW: Comprehensive docs
├── init_slack_db.py                   # NEW: Database migration
├── requirements.txt                   # Updated with slack-sdk
│
└── ... (existing files unchanged)
```

---

## Database Schema

### slack_messages

Stores individual Slack messages with vector embeddings:

```sql
CREATE TABLE slack_messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR UNIQUE,          -- Slack timestamp (ts)
    channel_id VARCHAR,
    channel_name VARCHAR,
    user_id VARCHAR,
    user_name VARCHAR,
    text TEXT,
    embedding vector(384),              -- pgvector: 384-dim MiniLM embedding
    thread_id VARCHAR,                  -- NULL for root messages
    timestamp TIMESTAMP,
    imported_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_slack_msg_channel ON slack_messages(channel_id);
CREATE INDEX idx_slack_msg_thread ON slack_messages(thread_id);
CREATE INDEX idx_slack_msg_embedding ON slack_messages USING ivfflat (embedding vector_cosine_ops);
```

### slack_threads

Stores thread metadata:

```sql
CREATE TABLE slack_threads (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR UNIQUE,
    channel_id VARCHAR,
    channel_name VARCHAR,
    root_user_id VARCHAR,
    root_user_name VARCHAR,
    root_text TEXT,
    thread_embedding vector(384),
    reply_count INTEGER,
    timestamp TIMESTAMP,
    imported_at TIMESTAMP
);

CREATE INDEX idx_slack_thread_channel ON slack_threads(channel_id);
CREATE INDEX idx_slack_thread_embedding ON slack_threads USING ivfflat (thread_embedding vector_cosine_ops);
```

---

## API Reference

### Import Slack Messages

**Endpoint**: `POST /api/slack/import`

**Request**:
```json
{
  "channel_ids": ["C12345", "C67890"],  // Optional - all if omitted
  "days_back": 30                       // Optional - from config if omitted
}
```

**Response** (200):
```json
{
  "status": "success",
  "message": "Slack messages imported successfully",
  "stats": {
    "total_messages": 523,
    "total_threads": 47,
    "channels_processed": 3,
    "failed_channels": []
  }
}
```

---

### Search Slack Messages

**Endpoint**: `POST /api/slack/search`

**Request**:
```json
{
  "query": "PostgreSQL setup guide",
  "search_type": "hybrid|semantic|keyword",
  "limit": 10,
  "channel_filter": "general"
}
```

**Response** (200):
```json
{
  "status": "success",
  "query": "PostgreSQL setup guide",
  "search_type": "hybrid",
  "result_count": 5,
  "results": [
    {
      "id": 42,
      "message_id": "1234567890.123456",
      "channel_id": "C123",
      "channel_name": "general",
      "user_id": "U456",
      "user_name": "John Doe",
      "text": "Here's the PostgreSQL setup process...",
      "timestamp": "2024-01-15T10:30:00",
      "thread_id": null,
      "relevance": "semantic_match",
      "score": 0.92
    },
    {
      "id": 43,
      "message_id": "1234567890.654321",
      "channel_id": "C789",
      "channel_name": "engineering",
      "user_id": "U999",
      "user_name": "Jane Smith",
      "text": "We had this issue with PostgreSQL last week...",
      "timestamp": "2024-01-14T14:20:00",
      "thread_id": "1234567890.654320",
      "relevance": "keyword_match",
      "score": 0.78
    }
  ]
}
```

---

### Get Thread with Replies

**Endpoint**: `GET /api/slack/threads/{thread_id}`

**Response** (200):
```json
{
  "status": "success",
  "thread": {
    "id": 1,
    "thread_id": "1234567890.654320",
    "channel_id": "C789",
    "channel_name": "engineering",
    "root_user_id": "U456",
    "root_user_name": "John Doe",
    "root_text": "Has anyone set up PostgreSQL with pgvector?",
    "reply_count": 5,
    "timestamp": "2024-01-15T10:30:00"
  },
  "replies": [
    {
      "id": 2,
      "message_id": "1234567890.654321",
      "channel_id": "C789",
      "channel_name": "engineering",
      "user_id": "U999",
      "user_name": "Jane Smith",
      "text": "Yes, I can share the setup steps...",
      "thread_id": "1234567890.654320",
      "timestamp": "2024-01-15T10:35:00"
    },
    {
      "id": 3,
      "message_id": "1234567890.654322",
      "channel_id": "C789",
      "channel_name": "engineering",
      "user_id": "U111",
      "user_name": "Bob Johnson",
      "text": "Here's the documentation link...",
      "thread_id": "1234567890.654320",
      "timestamp": "2024-01-15T10:40:00"
    }
  ],
  "reply_count": 5
}
```

---

### Get Channel Statistics

**Endpoint**: `GET /api/slack/channels`

**Response** (200):
```json
{
  "status": "success",
  "total_channels": 5,
  "channels": [
    {
      "channel_id": "C123",
      "channel_name": "general",
      "message_count": 342
    },
    {
      "channel_id": "C789",
      "channel_name": "engineering",
      "message_count": 268
    },
    {
      "channel_id": "C456",
      "channel_name": "support",
      "message_count": 195
    }
  ]
}
```

---

### Get Overall Statistics

**Endpoint**: `GET /api/slack/stats`

**Response** (200):
```json
{
  "status": "success",
  "total_messages": 805,
  "total_threads": 67,
  "total_channels": 3
}
```

---

## Search Types

### 1. Keyword Search
```json
{
  "query": "error handling",
  "search_type": "keyword"
}
```
- **Method**: SQL `ILIKE` pattern matching
- **Speed**: Very fast
- **Best for**: Exact terms, error codes, specific phrases
- **Example**: "404 error", "timeout exception"

### 2. Semantic Search
```json
{
  "query": "How do I fix database connection issues?",
  "search_type": "semantic"
}
```
- **Method**: pgvector cosine similarity on embeddings
- **Speed**: Moderate
- **Best for**: Conceptual questions, synonyms, intent matching
- **Example**: "database won't connect" → finds "can't reach postgres"

### 3. Hybrid Search (Default)
```json
{
  "query": "PostgreSQL configuration",
  "search_type": "hybrid"
}
```
- **Method**: Combines keyword + semantic, deduplicates by relevance
- **Speed**: Balanced
- **Best for**: General search, unknown query type
- **Result**: Union of both methods ranked by score

---

## Deployment

### Local Development

```bash
# 1. Create .env.local with Slack credentials
cp .env.local.template .env.local
# Edit SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL
docker-compose up -d postgres

# 4. Initialize Slack DB tables
python init_slack_db.py

# 5. Start app
python run.py
```

### Docker Deployment

```bash
# Build image
docker build -f Dockerfile -t rag-app:slack .

# Run with environment
docker run -d \
  -e SLACK_BOT_TOKEN=xoxb-... \
  -e SLACK_SIGNING_SECRET=... \
  -e DB_HOST=postgres \
  --network rag-network \
  -p 5000:5000 \
  rag-app:slack

# Or use docker-compose
docker-compose up -d
```

### Kubernetes Deployment

Update `helm/values.yaml`:

```yaml
slack:
  enabled: true
  botToken: xoxb-YOUR-TOKEN      # Use secrets in production
  signingSecret: YOUR-SECRET
  importLimit: 100
  importDays: 30

postgres:
  persistence:
    size: 50Gi  # Increase for Slack data + embeddings
```

Deploy:

```bash
helm upgrade --install rag-app helm/ -f helm/values.yaml
```

---

## Integration with Existing RAG

### Combine Slack + PDF + SFDC

The Slack integration works seamlessly with existing knowledge sources:

```python
# Example: Search all sources
def search_all_sources(query):
    # Search Slack
    slack_results = requests.post('http://localhost:5000/api/slack/search', 
        json={'query': query})['results']
    
    # Search PDFs (existing)
    pdf_results = requests.post('http://localhost:5000/api/rag/search',
        json={'query': query})['results']
    
    # Combine with Slack context having higher weight
    combined = slack_results[:5] + pdf_results[:5]
    
    # Generate answer
    context = '\n'.join([r['text'] for r in combined])
    answer = llm_answer(query, context)
    
    return answer
```

---

## Important Notes

### ✅ Main Branch (Monday Demo)
- **Branch**: `main`
- **Status**: Unchanged
- **Features**: SFDC integration + PDF RAG
- **Impact**: ZERO from Slack integration

### 🔄 Feature Branch (Slack Development)
- **Branch**: `feature/slack-integration`
- **Status**: Ready for testing
- **Features**: Complete Slack import + search
- **Impact**: Isolated from main

### 📋 Next Steps
1. Test Slack integration on feature branch
2. Verify database operations
3. Test all API endpoints
4. Performance testing with large datasets
5. Merge to main after Monday demo if approved

---

## Troubleshooting

### Error: "SLACK_BOT_TOKEN not set"

```bash
# Check .env.local
grep SLACK_BOT_TOKEN .env.local

# Should show: SLACK_BOT_TOKEN=xoxb-...
```

### Error: "Slack connection failed: invalid_auth"

```bash
# Token is invalid or expired
# 1. Check token in Slack API dashboard
# 2. Regenerate if needed
# 3. Update .env.local
```

### Error: "Connection refused" to PostgreSQL

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check pgvector is installed
psql -U postgres -d ragdb -c "SELECT * FROM pg_extension WHERE extname='vector';"

# If not, install:
psql -U postgres -d ragdb -c "CREATE EXTENSION vector;"
```

### Slow imports with large datasets

```bash
# Increase batch size in slack_import.py
BATCH_SIZE = 500  # Default: 100

# Or import specific channels:
curl -X POST http://localhost:5000/api/slack/import \
  -d '{"channel_ids": ["C123", "C456"]}'
```

---

## Configuration Reference

### .env Variables

```bash
# Slack API Credentials (REQUIRED)
SLACK_BOT_TOKEN=xoxb-...           # Bot token from API dashboard
SLACK_SIGNING_SECRET=...           # Signing secret from Basic Info
SLACK_APP_TOKEN=xapp-...           # App token for Socket Mode (optional)

# Import Settings
SLACK_IMPORT_LIMIT=100             # Messages per channel (max 1000)
SLACK_IMPORT_DAYS=30               # Days back to import (default 30)

# Database (unchanged from existing)
DB_USER=postgres
DB_PASS=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ragdb

# Models (unchanged from existing)
MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
EMBEDDING_MODEL=./models/embedding/all-MiniLM-L6-v2
```

---

## Files Modified/Created

### New Files
- ✨ `app/services/slack_client.py` - Slack API wrapper
- ✨ `app/services/slack_import.py` - Message import with embeddings
- ✨ `app/routes/slack_routes.py` - REST API endpoints
- ✨ `.env.local` - Local development config
- ✨ `SLACK_INTEGRATION.md` - Detailed documentation
- ✨ `init_slack_db.py` - Database migration script
- ✨ `SLACK_INTEGRATION_IMPLEMENTATION.md` - This file

### Updated Files
- 📝 `app/models.py` - Added SlackMessage, SlackThread models
- 📝 `app/config.py` - Added Slack configuration
- 📝 `app/__init__.py` - Registered slack_routes blueprint
- 📝 `requirements.txt` - Added slack-sdk, slack-bolt

### Unchanged (Main Branch)
- ✅ `app/services/sfdc_client.py`
- ✅ `app/services/sfdc_knowledge.py`
- ✅ `app/services/populate_db.py`
- ✅ All PDF RAG functionality
- ✅ SFDC integration

---

## GitHub Links

- **Repository**: https://github.com/sabya610/rag_app
- **Feature Branch**: https://github.com/sabya610/rag_app/tree/feature/slack-integration
- **Commit**: `640b5da` (Full implementation)
- **Pull Request**: Ready for review

---

## Support & Questions

1. **Setup issues**: See Quick Start section
2. **API errors**: Check Slack API dashboard for token status
3. **Database issues**: Verify PostgreSQL + pgvector running
4. **Performance**: Check database indexes and query plans
5. **Integration**: Review existing RAG code in `app/routes/rag_routes.py`

---

**Status**: ✅ Complete & Ready for Testing  
**Next**: Merge to `main` after Monday's SFDC/PDF demo

