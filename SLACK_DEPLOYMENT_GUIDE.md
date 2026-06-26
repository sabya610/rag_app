# Slack Integration - Deployment & Host Configuration

## Overview

This document provides deployment instructions for the Slack integration on your existing rag_app host/server. The integration uses the same infrastructure (PostgreSQL + pgvector) and can be deployed alongside the existing SFDC/PDF demo.

---

## Host Setup Requirements

### Prerequisites

Ensure your host has:

```bash
# 1. Docker & Docker Compose
docker --version      # v24.0+
docker-compose --version  # v2.0+

# 2. PostgreSQL with pgvector
# Already running from existing rag_app setup

# 3. Free ports
# 5000 - Flask app (if deploying in same container)
# 5432 - PostgreSQL (if separate service)
```

### Verify Existing Setup

```bash
# Check PostgreSQL running
docker-compose ps postgres

# Verify pgvector extension
docker-compose exec postgres psql -U postgres -d ragdb -c "SELECT * FROM pg_extension;"

# Output should include pgvector
```

---

## Deployment Options

### Option 1: Local Development (Single Host)

**Best for**: Testing and validation

```bash
# 1. Clone/pull latest code
cd rag_app
git fetch origin
git checkout feature/slack-integration

# 2. Create .env.local
cat > .env.local << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ragdb
MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
EMBEDDING_MODEL=./models/embedding/all-MiniLM-L6-v2
SLACK_BOT_TOKEN=xoxb-YOUR-TOKEN
SLACK_SIGNING_SECRET=YOUR-SECRET
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30
EOF

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python init_slack_db.py

# 5. Start Flask app
python run.py

# 6. Test endpoints (in new terminal)
curl http://localhost:5000/api/slack/stats
```

---

### Option 2: Docker Container Deployment

**Best for**: Production-like environment

#### Build Image

```bash
# Use existing Dockerfile (already supports slack-sdk)
docker build -f Dockerfile -t rag-app:slack-latest .

# Or with tag
docker build -f Dockerfile -t your-registry.azurecr.io/rag-app:slack-v1.0 .
```

#### Run Container

```bash
# Create environment file
cat > docker.env << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_HOST=postgres-service    # Docker service name
DB_PORT=5432
DB_NAME=ragdb
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30
EOF

# Run container
docker run -d \
  --name rag-app-slack \
  --env-file docker.env \
  -p 5000:5000 \
  --network rag-network \
  rag-app:slack-latest

# Verify running
docker logs rag-app-slack
docker ps | grep rag-app-slack
```

---

### Option 3: Docker Compose Deployment

**Best for**: Multi-service orchestration

#### Update docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:latest
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - rag-network

  rag-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      DB_USER: postgres
      DB_PASS: postgres
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: ragdb
      SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN}
      SLACK_SIGNING_SECRET: ${SLACK_SIGNING_SECRET}
      SLACK_IMPORT_LIMIT: 100
      SLACK_IMPORT_DAYS: 30
    depends_on:
      - postgres
    networks:
      - rag-network
    volumes:
      - ./models:/app/rag_app/models:ro
      - ./app:/app/rag_app/app

volumes:
  postgres_data:

networks:
  rag-network:
    driver: bridge
```

#### Deploy

```bash
# Create .env for docker-compose
cat > .env.docker << 'EOF'
SLACK_BOT_TOKEN=xoxb-YOUR-TOKEN
SLACK_SIGNING_SECRET=YOUR-SECRET
EOF

# Start services
docker-compose --env-file .env.docker up -d

# Check logs
docker-compose logs -f rag-app

# Stop services
docker-compose down
```

---

### Option 4: Kubernetes Deployment

**Best for**: Production enterprise deployment

#### Create ConfigMap for Slack Config

```bash
kubectl create configmap slack-config \
  --from-literal=SLACK_IMPORT_LIMIT=100 \
  --from-literal=SLACK_IMPORT_DAYS=30 \
  -n rag-app

# Create secret for credentials
kubectl create secret generic slack-credentials \
  --from-literal=SLACK_BOT_TOKEN=xoxb-YOUR-TOKEN \
  --from-literal=SLACK_SIGNING_SECRET=YOUR-SECRET \
  -n rag-app
```

#### Update Helm Values

```yaml
# helm/values.yaml
slack:
  enabled: true
  importLimit: 100
  importDays: 30
  credentials:
    botTokenRef:
      name: slack-credentials
      key: SLACK_BOT_TOKEN
    signingSecretRef:
      name: slack-credentials
      key: SLACK_SIGNING_SECRET

postgres:
  enabled: true
  persistence:
    size: 100Gi  # Larger due to Slack embeddings
  settings:
    shared_preload_libraries: vector
```

#### Deploy to K8s

```bash
# Install or upgrade
helm upgrade --install rag-app helm/ \
  -f helm/values.yaml \
  -n rag-app \
  --create-namespace

# Verify deployment
kubectl get pods -n rag-app
kubectl get svc -n rag-app

# Check app logs
kubectl logs -n rag-app -l app=rag-app -f

# Port forward for testing
kubectl port-forward -n rag-app svc/rag-app 5000:5000
```

---

## Post-Deployment Setup

### 1. Initialize Database Tables

After deployment, create Slack tables:

```bash
# For local/Docker
python init_slack_db.py

# For Kubernetes
kubectl exec -n rag-app deployment/rag-app -- python init_slack_db.py
```

### 2. Import Slack Messages

```bash
# First import from all channels
curl -X POST http://localhost:5000/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{"days_back": 30}'

# Or specific channels
curl -X POST http://localhost:5000/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{
    "channel_ids": ["C123", "C456"],
    "days_back": 30
  }'
```

### 3. Verify Deployment

```bash
# Check stats
curl http://localhost:5000/api/slack/stats

# Should output:
# {
#   "status": "success",
#   "total_messages": 500,
#   "total_threads": 45,
#   "total_channels": 3
# }

# Test search
curl -X POST http://localhost:5000/api/slack/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 5}'
```

---

## Host Configuration Examples

### AWS EC2

```bash
# 1. SSH into instance
ssh -i key.pem ec2-user@your-instance.compute.amazonaws.com

# 2. Install Docker
amazon-linux-extras install docker -y
systemctl start docker

# 3. Clone repo
git clone https://github.com/sabya610/rag_app.git
cd rag_app
git checkout feature/slack-integration

# 4. Configure environment
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...

# 5. Deploy
docker-compose up -d

# 6. Check running
curl http://localhost:5000/api/slack/stats
```

### Azure Container Instance (ACI)

```bash
# Create resource group
az group create --name rag-app-rg --location eastus

# Create environment file
cat > env.list << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_HOST=postgres.postgres.svc.cluster.local
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
EOF

# Deploy container
az container create \
  --resource-group rag-app-rg \
  --name rag-app-slack \
  --image rag-app:slack-latest \
  --ports 5000 \
  --environment-variables-from-file env.list

# Get IP
az container show \
  --resource-group rag-app-rg \
  --name rag-app-slack \
  --query ipAddress.ip
```

### On-Premise / VMware

```bash
# 1. SSH to VM
ssh admin@vm-hostname

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Join user to docker group
sudo usermod -aG docker $USER
newgrp docker

# 4. Clone repo
cd /opt
git clone https://github.com/sabya610/rag_app.git
cd rag_app
git checkout feature/slack-integration

# 5. Create systemd service
sudo tee /etc/systemd/system/rag-app.service > /dev/null <<EOF
[Unit]
Description=RAG App with Slack Integration
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/rag_app
EnvironmentFile=/opt/rag_app/.env.local
ExecStart=/usr/bin/docker-compose up
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 6. Start service
sudo systemctl daemon-reload
sudo systemctl enable rag-app
sudo systemctl start rag-app
sudo systemctl status rag-app

# 7. View logs
journalctl -u rag-app -f
```

---

## Security Configuration

### Environment Variables (Never Hardcode)

```bash
# ✅ Correct: Use .env file
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...

# ✅ Or use secrets manager
# AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id rag-app/slack

# Azure Key Vault
az keyvault secret show --vault-name rag-app-kv --name slack-bot-token

# Docker Secrets (Swarm)
docker secret create slack_token -
```

### Network Security

```bash
# Configure firewall rules
# Allow: 5000 (Flask app) - only from authorized IPs
# Allow: 5432 (PostgreSQL) - only from Flask container

# For Linux
sudo ufw allow from 10.0.0.0/8 to any port 5000
sudo ufw allow from 172.17.0.0/16 to any port 5432
```

### SSL/TLS Configuration

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Configure Flask for HTTPS
# In app/__init__.py or config.py
# app.run(ssl_context=('cert.pem', 'key.pem'))

# Or use nginx as reverse proxy
# nginx.conf: 443 → 5000 (localhost)
```

---

## Performance Tuning

### PostgreSQL Configuration

```bash
# For large Slack datasets, optimize PostgreSQL
# In postgresql.conf:

shared_buffers = 4GB                    # 25% of RAM
effective_cache_size = 12GB             # 75% of RAM
maintenance_work_mem = 1GB
max_parallel_workers_per_gather = 4
random_page_cost = 1.1                  # For SSDs
effective_io_concurrency = 200          # For SSDs
work_mem = 50MB
wal_buffers = 16MB
default_statistics_target = 100
log_min_duration_statement = 1000       # Log slow queries
```

### Vector Index Configuration

```sql
-- For faster semantic search
-- Use IVFFlat for medium datasets (< 1M vectors)
CREATE INDEX idx_slack_embedding_ivf 
ON slack_messages USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Use HNSW for large datasets (> 1M vectors)
CREATE INDEX idx_slack_embedding_hnsw 
ON slack_messages USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Application Tuning

```python
# In config.py
MAX_RESULTS = 50            # Retrieve top 50 results
BATCH_SIZE = 500           # Import in batches of 500
CACHE_TTL = 3600           # Cache search results 1 hour
POOL_SIZE = 20             # DB connection pool
POOL_RECYCLE = 3600        # Recycle connections
```

---

## Monitoring & Logging

### Application Metrics

```bash
# Check import progress
curl http://localhost:5000/api/slack/stats

# Database size
docker exec postgres psql -U postgres -d ragdb -c \
  "SELECT pg_size_pretty(pg_database_size('ragdb'));"

# Vector index size
docker exec postgres psql -U postgres -d ragdb -c \
  "SELECT pg_size_pretty(pg_indexes_size('slack_messages'));"
```

### Application Logs

```bash
# Docker logs
docker logs rag-app-slack -f

# Docker Compose
docker-compose logs -f rag-app

# Kubernetes
kubectl logs -n rag-app deployment/rag-app -f

# From Python
# Already configured in app/__init__.py
# Logs go to stdout → Docker/Kubernetes captures them
```

### Database Monitoring

```sql
-- Check active queries
SELECT pid, query, state FROM pg_stat_activity;

-- Check pgvector index stats
SELECT * FROM pg_stat_user_indexes 
WHERE relname = 'idx_slack_msg_embedding';

-- Check slow queries
SELECT query, calls, mean_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
```

---

## Troubleshooting Deployment

### Issue: "Connection refused" to PostgreSQL

```bash
# 1. Verify PostgreSQL is running
docker ps | grep postgres

# 2. Check network connectivity
docker network inspect rag-network
# Should show both postgres and rag-app containers

# 3. Test connectivity from app container
docker-compose exec rag-app \
  python -c "import psycopg2; psycopg2.connect('postgresql://...')"

# 4. Check credentials in .env
grep DB_ .env.local
```

### Issue: "Slack token invalid"

```bash
# 1. Verify token format
grep SLACK_BOT_TOKEN .env.local
# Should start with xoxb-

# 2. Test token directly
curl -X POST https://slack.com/api/auth.test \
  -H "Authorization: Bearer xoxb-YOUR-TOKEN"

# 3. If expired, regenerate from https://api.slack.com/apps
# 4. Update .env and restart container
```

### Issue: "Slow import"

```bash
# Monitor progress
docker logs rag-app-slack -f | grep "Fetched\|Imported"

# Check database load
docker exec postgres top -b -n 1 | head -20

# Reduce import batch size in slack_import.py
# Reduce SLACK_IMPORT_LIMIT in .env
```

### Issue: "High memory usage"

```bash
# Check memory limits
docker stats rag-app-slack

# Set memory limit
docker update --memory 4g rag-app-slack

# Or in docker-compose.yml
# services:
#   rag-app:
#     mem_limit: 4g
```

---

## Backup & Recovery

### Database Backup

```bash
# Full backup
docker exec postgres pg_dump -U postgres ragdb | \
  gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Restore from backup
gunzip < backup_20240115_120000.sql.gz | \
  docker exec -i postgres psql -U postgres -d ragdb
```

### Persistent Volumes

```bash
# Docker Compose
volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres

# Kubernetes
spec:
  persistentVolumeClaim:
    claimName: postgres-pvc
```

---

## Scaling Considerations

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Balancing

```bash
# Nginx as reverse proxy
upstream rag_app {
    server rag-app-1:5000;
    server rag-app-2:5000;
    server rag-app-3:5000;
}

server {
    listen 80;
    location / {
        proxy_pass http://rag_app;
    }
}
```

---

## Maintenance

### Regular Tasks

```bash
# Daily: Check logs for errors
docker logs rag-app-slack | grep ERROR

# Weekly: Verify backups
ls -lh backup_*.sql.gz | tail -5

# Monthly: Analyze database
docker exec postgres vacuumdb -U postgres -a

# Monthly: Update Docker images
docker pull rag-app:slack-latest
docker-compose pull && docker-compose up -d
```

---

## Summary

| Deployment | Best For | Complexity | Cost |
|------------|----------|-----------|------|
| Local Dev | Testing | Low | Free |
| Docker | Single server | Low | Low |
| K8s | Enterprise | High | Medium |
| Cloud (AWS/Azure) | Scalability | Medium | Medium-High |

**Recommended**: Start with Docker Compose, move to K8s for production.

---

## Support Resources

- Slack Integration Docs: `SLACK_INTEGRATION.md`
- Implementation Guide: `SLACK_INTEGRATION_IMPLEMENTATION.md`
- GitHub Branch: `feature/slack-integration`
- Python Slack SDK: https://slack.dev/python-slack-sdk/
- PostgreSQL pgvector: https://github.com/pgvector/pgvector

