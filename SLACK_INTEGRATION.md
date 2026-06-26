# Slack Integration for rag_app

## Overview

This document describes the Slack integration for the rag_app RAG (Retrieval-Augmented Generation) system. The integration allows you to:

1. **Import Slack messages** from channels into the PostgreSQL database
2. **Generate embeddings** for semantic search using pgvector
3. **Search Slack data** using keyword search, semantic search, or hybrid approach
4. **Answer questions** about Slack conversation history using the LLM

## Architecture

```
Slack Workspace
      │
      ├── Slack API (Bot Token)
      │
Flask Application
      │
      ├── Slack Import Service
      │        ├── Fetch channels
      │        ├── Fetch messages & threads
      │        └── Generate embeddings
      │
      ├── PostgreSQL Database
      │        ├── slack_messages (text + pgvector embedding)
      │        ├── slack_threads (thread metadata + embedding)
      │        └── kb_chunks (original PDF/KB data)
      │
      ├── Search Engine
      │        ├── Keyword Search (SQL ILIKE)
      │        ├── Semantic Search (pgvector cosine similarity)
      │        └── Hybrid Search (combined)
      │
      └── LLM (Llama.cpp)
               └── Answer generation with Slack context
```

## Setup Instructions

### Step 1: Create Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. App name: `rag-app`
4. Pick your workspace

### Step 2: Configure Bot Token Scopes

In the app settings, go to **OAuth & Permissions** and add these scopes:

```
channels:history
channels:read
users:read
users:read.email
chat:write
```

### Step 3: Install App to Workspace

1. Under **OAuth & Permissions**, click "Install to Workspace"
2. Authorize the app
3. Copy your **Bot User OAuth Token** (starts with `xoxb-`)

### Step 4: Get Signing Secret

1. Go to **Basic Information**
2. Copy your **Signing Secret**

### Step 5: Enable Socket Mode (Optional)

For real-time Slack events:
1. Go to **Socket Mode** and toggle it on
2. Generate a new token (starts with `xapp-`)

### Step 6: Configure rag_app

Update your `.env.local`:

```bash
SLACK_BOT_TOKEN=xoxb-YOUR-BOT-TOKEN
SLACK_SIGNING_SECRET=YOUR-SIGNING-SECRET
SLACK_APP_TOKEN=xapp-YOUR-APP-TOKEN
SLACK_IMPORT_LIMIT=100        # Messages per channel
SLACK_IMPORT_DAYS=30          # How far back to import
```

### Step 7: Create Database Tables

The tables are created automatically on app startup:

- `slack_messages` - Individual messages with embeddings
- `slack_threads` - Thread metadata

## API Endpoints

### Import Slack Messages

```http
POST /api/slack/import
Content-Type: application/json

{
  "channel_ids": ["C123", "C456"],  // Optional - all channels if omitted
  "days_back": 30                   // Optional - uses SLACK_IMPORT_DAYS default
}

Response:
{
  "status": "success",
  "stats": {
    "total_messages": 500,
    "total_threads": 50,
    "channels_processed": 3,
    "failed_channels": []
  }
}
```

### Search Slack Messages

```http
POST /api/slack/search
Content-Type: application/json

{
  "query": "help with setup",
  "search_type": "semantic|keyword|hybrid",  // Default: hybrid
  "limit": 10,
  "channel_filter": "general"               // Optional
}

Response:
{
  "status": "success",
  "query": "help with setup",
  "search_type": "hybrid",
  "result_count": 10,
  "results": [
    {
      "id": 1,
      "message_id": "1234567890.123456",
      "channel_name": "general",
      "user_name": "John Doe",
      "text": "Here's how to set up...",
      "timestamp": "2024-01-15T10:30:00",
      "relevance": "semantic_match",
      "score": 0.95
    },
    ...
  ]
}
```

### Get Thread with Replies

```http
GET /api/slack/threads/{thread_id}

Response:
{
  "status": "success",
  "thread": {
    "thread_id": "1234567890.123456",
    "channel_name": "general",
    "root_user_name": "John Doe",
    "root_text": "Question about setup",
    "reply_count": 5,
    "timestamp": "2024-01-15T10:30:00"
  },
  "replies": [
    {
      "message_id": "1234567890.654321",
      "user_name": "Jane Smith",
      "text": "Here's the answer...",
      "timestamp": "2024-01-15T10:35:00"
    },
    ...
  ],
  "reply_count": 5
}
```

### Get Channel Statistics

```http
GET /api/slack/channels

Response:
{
  "status": "success",
  "total_channels": 3,
  "channels": [
    {
      "channel_id": "C123",
      "channel_name": "general",
      "message_count": 250
    },
    {
      "channel_id": "C456",
      "channel_name": "random",
      "message_count": 180
    }
  ]
}
```

### Get Overall Statistics

```http
GET /api/slack/stats

Response:
{
  "status": "success",
  "total_messages": 430,
  "total_threads": 45,
  "total_channels": 2
}
```

## Search Types Explained

### Keyword Search
- Uses SQL `ILIKE` for full-text pattern matching
- Fast, good for exact or partial word matches
- Example: "setup", "error message"

### Semantic Search
- Uses pgvector cosine similarity on embeddings
- Understands meaning, better for conceptual queries
- Slower but more accurate for understanding intent
- Example: "how do I configure the application?"

### Hybrid Search
- Combines both keyword and semantic results
- Deduplicates and sorts by relevance score
- Best for general use cases

## Workflow Example

### 1. Import Slack Data

```bash
curl -X POST http://localhost:5000/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{
    "channel_ids": ["C123"],
    "days_back": 30
  }'
```

### 2. Search for Information

```bash
curl -X POST http://localhost:5000/api/slack/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to configure PostgreSQL",
    "search_type": "hybrid",
    "limit": 5
  }'
```

### 3. Get Full Thread Context

```bash
curl http://localhost:5000/api/slack/threads/1234567890.123456
```

### 4. Generate LLM Answer

Use the search results as context for the existing RAG endpoint:

```bash
curl -X POST http://localhost:5000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "help me configure PostgreSQL",
    "include_slack": true
  }'
```

## Database Schema

### slack_messages table

```sql
CREATE TABLE slack_messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR UNIQUE NOT NULL,     -- Slack timestamp
    channel_id VARCHAR NOT NULL,
    channel_name VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    user_name VARCHAR NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384),                  -- pgvector embedding
    thread_id VARCHAR,                      -- NULL for root messages
    timestamp TIMESTAMP NOT NULL,
    imported_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_slack_msg_channel ON slack_messages(channel_id);
CREATE INDEX idx_slack_msg_thread ON slack_messages(thread_id);
CREATE INDEX idx_slack_msg_embedding ON slack_messages USING ivfflat (embedding vector_cosine_ops);
```

### slack_threads table

```sql
CREATE TABLE slack_threads (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR UNIQUE NOT NULL,
    channel_id VARCHAR NOT NULL,
    channel_name VARCHAR NOT NULL,
    root_user_id VARCHAR NOT NULL,
    root_user_name VARCHAR NOT NULL,
    root_text TEXT NOT NULL,
    thread_embedding vector(384),          -- pgvector embedding
    reply_count INTEGER DEFAULT 0,
    timestamp TIMESTAMP NOT NULL,
    imported_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_slack_thread_channel ON slack_threads(channel_id);
CREATE INDEX idx_slack_thread_embedding ON slack_threads USING ivfflat (thread_embedding vector_cosine_ops);
```

## Running Locally

### Prerequisites

```bash
# Python dependencies
pip install -r requirements.txt

# PostgreSQL running with pgvector extension
docker-compose up -d postgres

# Create pgvector extension
psql -U postgres -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Start the App

```bash
# Load environment variables
source .env.local

# Run Flask development server
python run.py

# App runs on http://localhost:5000
```

## Deployment Notes

### Kubernetes Deployment

The Slack integration follows the same deployment pattern as existing rag_app components:

1. Database migrations run on startup
2. Tables are created automatically
3. No manual schema setup required
4. Connection pooling works with pgvector

### Environment Variables for Production

Update your deployment `.env`:

```bash
SLACK_BOT_TOKEN=xoxb-...           # From Slack API dashboard
SLACK_SIGNING_SECRET=...           # From Slack API dashboard  
SLACK_APP_TOKEN=xapp-...           # Optional for Socket Mode
DB_HOST=postgres-service           # K8s service name
DB_PORT=5432
DB_NAME=ragdb
```

### Performance Considerations

- **Large imports**: Import in batches, monitor database size
- **Search performance**: Ensure pgvector indexes are created
- **Embedding generation**: CPU-intensive, may take time for large datasets
- **Memory**: Keep sentence-transformers model in memory

## Troubleshooting

### Issue: "SLACK_BOT_TOKEN not set"

**Solution**: Ensure `.env.local` has valid `SLACK_BOT_TOKEN`

```bash
grep SLACK_BOT_TOKEN .env.local
```

### Issue: "Failed to connect to Slack API"

**Possible causes**:
- Token is invalid or expired
- Bot is not invited to channels
- Network connectivity issue

**Solution**:
```bash
# Test Slack connection
curl -s https://slack.com/api/auth.test \
  -H "Authorization: Bearer xoxb-YOUR-TOKEN"
```

### Issue: Slow search performance

**Solution**: Ensure pgvector indexes are created

```sql
-- Check indexes
SELECT * FROM pg_indexes WHERE tablename = 'slack_messages';

-- Create index if missing
CREATE INDEX idx_slack_msg_embedding ON slack_messages 
  USING ivfflat (embedding vector_cosine_ops);
```

### Issue: Duplicate messages during re-import

**Solution**: The system checks for duplicate `message_id` before importing. Clear the table to re-import:

```sql
TRUNCATE slack_messages, slack_threads;
```

## Integration with Main RAG Pipeline

The Slack integration can be combined with PDF/SFDC knowledge:

1. **Multi-source search**: Search both Slack and PDF/SFDC data
2. **Combined context**: Use Slack discussions + official documentation
3. **Unified interface**: Single API for all knowledge sources

Example combining sources:

```python
# Search Slack
slack_results = search_slack_messages("question")

# Search PDFs
pdf_results = search_kb_chunks("question")

# Combine context for LLM
combined_context = slack_results + pdf_results
answer = generate_answer_with_context("question", combined_context)
```

## Next Steps

1. ✅ Create Slack App and get tokens
2. ✅ Configure `.env.local` with credentials
3. ✅ Start the application
4. ✅ Import Slack messages via `/api/slack/import`
5. ✅ Test search via `/api/slack/search`
6. ✅ Generate answers using full RAG pipeline

## Support

For issues or questions:
1. Check logs: `docker logs rag-app`
2. Test Slack API: https://api.slack.com/apps
3. Review database: `psql -U postgres -d ragdb`
4. Check pgvector status: `SELECT * FROM pg_extension;`

