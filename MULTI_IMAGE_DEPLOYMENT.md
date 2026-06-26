# Multi-Image Deployment Strategy

**Purpose**: Maintain separate Docker images for stable SFDC/PDF demo and new Slack integration, with easy switching and rollback capabilities.

---

## Overview

| Aspect | SFDC + PDF | SFDC + PDF + Slack |
|--------|-----------|-------------------|
| **Image** | `rag-app:1.0-sfdc-pdf` | `rag-app:1.0-slack-integrated` |
| **Dockerfile** | `Dockerfile.sfdc-pdf` | `Dockerfile.slack-integrated` |
| **Compose** | `docker-compose.sfdc-pdf.yml` | `docker-compose.slack-integrated.yml` |
| **Purpose** | Production-stable (Monday Demo) | Feature development/testing |
| **Port** | 5000 | 5001 |
| **DB Port** | 5432 | 5433 |
| **Branch** | `main` | `feature/slack-integration` |
| **Slack** | ❌ Disabled | ✅ Enabled |
| **Status** | ✅ Production Ready | 🚀 Testing Phase |

---

## Quick Start: Switch Between Images

### Deploy SFDC + PDF (Monday Demo - Stable)

```bash
# Switch to main branch (stable)
cd /opt/rag_app
git checkout main

# Create .env for demo
cat > .env.demo-sfdc << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_NAME=ragdb
MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
EMBEDDING_MODEL=./models/embedding/all-MiniLM-L6-v2
SFDC_USERNAME=your-sfdc-user
SFDC_PASSWORD=your-sfdc-pass
SFDC_SECURITY_TOKEN=your-token
SFDC_CLIENT_ID=your-client-id
SFDC_CLIENT_SECRET=your-client-secret
EOF

# Deploy SFDC + PDF stack
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d

# Verify running
docker ps | grep rag-app-sfdc-pdf
curl http://localhost:5000/api/rag/stats
```

**Result**:
- ✅ App running on `http://localhost:5000`
- ✅ PostgreSQL on `localhost:5432`
- ✅ SFDC + PDF integration active
- ✅ Slack disabled

---

### Deploy SFDC + PDF + Slack (Feature Testing)

```bash
# Switch to feature branch
cd /opt/rag_app
git checkout feature/slack-integration

# Create .env for slack integration testing
cat > .env.test-slack << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_NAME=ragdb
MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
EMBEDDING_MODEL=./models/embedding/all-MiniLM-L6-v2
# SFDC
SFDC_USERNAME=your-sfdc-user
SFDC_PASSWORD=your-sfdc-pass
SFDC_SECURITY_TOKEN=your-token
SFDC_CLIENT_ID=your-client-id
SFDC_CLIENT_SECRET=your-client-secret
# Slack
SLACK_BOT_TOKEN=xoxb-YOUR-TOKEN
SLACK_SIGNING_SECRET=YOUR-SECRET
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30
EOF

# Deploy Slack-integrated stack
docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d

# Initialize Slack database
docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py

# Verify running
docker ps | grep rag-app-slack-integrated
curl http://localhost:5001/api/slack/stats
```

**Result**:
- ✅ App running on `http://localhost:5001` (different port)
- ✅ PostgreSQL on `localhost:5433` (different port)
- ✅ SFDC + PDF + Slack integration active
- ✅ Separate database from SFDC/PDF instance

---

## Running Both Simultaneously

**Use Case**: Test Slack integration while keeping Monday demo running

```bash
# Terminal 1: SFDC + PDF Demo
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml up -d

# Terminal 2: Slack Integration Testing
git checkout feature/slack-integration
docker-compose -f docker-compose.slack-integrated.yml up -d

# Check both running
docker ps | grep rag-app
# Output should show both containers
```

**Endpoints**:
- SFDC + PDF: `http://localhost:5000/api/rag/stats`
- Slack: `http://localhost:5001/api/slack/stats`

---

## Switching Between Configurations

### Switch from Slack Back to SFDC + PDF

```bash
# Stop Slack container
docker-compose -f docker-compose.slack-integrated.yml down

# Switch to main branch
git checkout main

# Start SFDC + PDF demo
docker-compose -f docker-compose.sfdc-pdf.yml up -d

# Verify
curl http://localhost:5000/api/rag/stats
```

### Switch from SFDC + PDF to Slack

```bash
# Stop SFDC demo
docker-compose -f docker-compose.sfdc-pdf.yml down

# Switch to feature branch
git checkout feature/slack-integration

# Start Slack integration
docker-compose -f docker-compose.slack-integrated.yml up -d

# Initialize DB
docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py

# Verify
curl http://localhost:5001/api/slack/stats
```

---

## Image Building & Tagging

### Build Images Locally

```bash
# Build SFDC + PDF image (main branch)
git checkout main
docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .
docker tag rag-app:1.0-sfdc-pdf rag-app:latest-stable

# Build Slack-integrated image (feature branch)
git checkout feature/slack-integration
docker build -f Dockerfile.slack-integrated -t rag-app:1.0-slack-integrated .
docker tag rag-app:1.0-slack-integrated rag-app:latest-development
```

### Push to Docker Registry (Azure ACR)

```bash
# Login to registry
az acr login --name your-registry

# Tag images for registry
docker tag rag-app:1.0-sfdc-pdf your-registry.azurecr.io/rag-app:1.0-sfdc-pdf
docker tag rag-app:1.0-slack-integrated your-registry.azurecr.io/rag-app:1.0-slack-integrated

# Push to registry
docker push your-registry.azurecr.io/rag-app:1.0-sfdc-pdf
docker push your-registry.azurecr.io/rag-app:1.0-slack-integrated

# Verify pushed
az acr repository list --name your-registry
az acr repository show-tags --name your-registry --repository rag-app
```

---

## Docker Compose Comparison

### SFDC + PDF Stack (docker-compose.sfdc-pdf.yml)

```yaml
Services:
  - postgres-sfdc-pdf (port 5432)
  - rag-app-sfdc-pdf (port 5000)

Environment:
  - DB_HOST: postgres-sfdc-pdf
  - FEATURES_SLACK: disabled
  - SFDC credentials required

Network:
  - rag-network

Volume:
  - postgres_sfdc_pdf_data
```

### Slack-Integrated Stack (docker-compose.slack-integrated.yml)

```yaml
Services:
  - postgres-slack-integrated (port 5433)
  - rag-app-slack-integrated (port 5001)

Environment:
  - DB_HOST: postgres-slack-integrated
  - FEATURES_SLACK: enabled
  - SFDC credentials + Slack credentials

Network:
  - rag-network-slack (isolated)

Volume:
  - postgres_slack_integrated_data
```

---

## Database Isolation

### Separate Databases

- **SFDC + PDF**: `ragdb` on `localhost:5432`
- **Slack**: `ragdb` on `localhost:5433`

Each uses its own PostgreSQL instance with independent data.

### Backup Strategy

```bash
# Backup SFDC + PDF database
docker exec postgres-sfdc-pdf pg_dump -U postgres ragdb | \
  gzip > backup_sfdc_pdf_$(date +%Y%m%d_%H%M%S).sql.gz

# Backup Slack database
docker exec postgres-slack-integrated pg_dump -U postgres ragdb | \
  gzip > backup_slack_$(date +%Y%m%d_%H%M%S).sql.gz

# Restore SFDC + PDF
gunzip < backup_sfdc_pdf_*.sql.gz | \
  docker exec -i postgres-sfdc-pdf psql -U postgres -d ragdb

# Restore Slack
gunzip < backup_slack_*.sql.gz | \
  docker exec -i postgres-slack-integrated psql -U postgres -d ragdb
```

---

## Deployment Scenarios

### Scenario 1: Monday Demo (Production)

**Timeline**: Monday 10:00 AM  
**Image**: `rag-app:1.0-sfdc-pdf`  
**Stack**: `docker-compose.sfdc-pdf.yml`

```bash
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml up -d
curl http://localhost:5000/api/rag/stats
```

### Scenario 2: Post-Demo Testing

**Timeline**: Monday 5:00 PM  
**Image**: `rag-app:1.0-slack-integrated`  
**Stack**: `docker-compose.slack-integrated.yml`

```bash
# Stop demo
docker-compose -f docker-compose.sfdc-pdf.yml down

# Start testing
git checkout feature/slack-integration
docker-compose -f docker-compose.slack-integrated.yml up -d
docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py
curl http://localhost:5001/api/slack/stats
```

### Scenario 3: Both Running (Parallel Testing)

**Timeline**: Tuesday  
**Images**: Both `1.0-sfdc-pdf` and `1.0-slack-integrated`

```bash
# Terminal 1
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml up

# Terminal 2
git checkout feature/slack-integration
docker-compose -f docker-compose.slack-integrated.yml up
```

---

## Rollback Procedures

### Quick Rollback to SFDC + PDF

```bash
# If Slack integration has issues, revert immediately:

# Stop current stack
docker-compose down

# Switch to main branch
git checkout main

# Deploy stable image
docker-compose -f docker-compose.sfdc-pdf.yml up -d

# Verify
curl http://localhost:5000/api/rag/stats
```

### Rollback Commands

```bash
# Kill all rag-app containers
docker ps | grep rag-app | awk '{print $1}' | xargs docker kill

# Remove Slack-specific containers
docker rm postgres-slack-integrated rag-app-slack-integrated

# Clean up Slack networks
docker network rm rag-network-slack

# Start stable SFDC + PDF stack
docker-compose -f docker-compose.sfdc-pdf.yml up -d
```

---

## Configuration Management

### Environment Variables (.env files)

```bash
# .env.demo-sfdc (Monday Demo)
# Contains only SFDC credentials
# Slack variables not needed

# .env.test-slack (Slack Testing)
# Contains SFDC + Slack credentials
# Both sets of credentials required
```

### Secret Management (Production)

```bash
# Store in environment, not in .env
export SLACK_BOT_TOKEN=xoxb-...
export SFDC_PASSWORD=...

# Or use Docker secrets (Swarm)
echo "xoxb-..." | docker secret create slack_token -
echo "password" | docker secret create sfdc_password -

# Or use environment variable files
docker run --env-file /run/secrets/env.list ...
```

---

## Monitoring & Health Checks

### Check SFDC + PDF Status

```bash
# Health endpoint
curl http://localhost:5000/api/rag/stats

# Response should show:
# - total_chunks
# - total_questions
# - total_sfdc_articles
```

### Check Slack Status

```bash
# Health endpoint
curl http://localhost:5001/api/slack/stats

# Response should show:
# - total_messages
# - total_threads
# - total_channels
```

### Container Health

```bash
# Check SFDC + PDF
docker ps --filter "name=rag-app-sfdc-pdf"
docker logs rag-app-sfdc-pdf | tail -20

# Check Slack
docker ps --filter "name=rag-app-slack-integrated"
docker logs rag-app-slack-integrated | tail -20
```

---

## Kubernetes Deployment

### Deploy SFDC + PDF

```yaml
# helm/values-sfdc-pdf.yaml
image:
  tag: "1.0-sfdc-pdf"
slack:
  enabled: false
sfdc:
  enabled: true
```

```bash
helm upgrade --install rag-app-demo helm/ \
  -f helm/values-sfdc-pdf.yaml \
  -n rag-app-demo
```

### Deploy Slack-Integrated

```yaml
# helm/values-slack-integrated.yaml
image:
  tag: "1.0-slack-integrated"
slack:
  enabled: true
sfdc:
  enabled: true
```

```bash
helm upgrade --install rag-app-slack helm/ \
  -f helm/values-slack-integrated.yaml \
  -n rag-app-slack
```

### Running Both

```bash
# Deploy both simultaneously
helm upgrade --install rag-app-demo helm/ -f helm/values-sfdc-pdf.yaml -n rag-app-demo
helm upgrade --install rag-app-slack helm/ -f helm/values-slack-integrated.yaml -n rag-app-slack

# Check both running
kubectl get deployments -n rag-app-demo
kubectl get deployments -n rag-app-slack
```

---

## CI/CD Pipeline Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/build-images.yml
name: Build Docker Images

on:
  push:
    branches: [main, feature/slack-integration]

jobs:
  build-sfdc-pdf:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build SFDC + PDF image
        run: |
          docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .
          docker push your-registry.azurecr.io/rag-app:1.0-sfdc-pdf

  build-slack:
    if: github.ref == 'refs/heads/feature/slack-integration'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Slack-integrated image
        run: |
          docker build -f Dockerfile.slack-integrated -t rag-app:1.0-slack-integrated .
          docker push your-registry.azurecr.io/rag-app:1.0-slack-integrated
```

---

## Tagging Strategy

### Git Tags for Releases

```bash
# Tag stable SFDC + PDF release
git tag -a v1.0-sfdc-pdf -m "SFDC + PDF - Monday Demo"
git push origin v1.0-sfdc-pdf

# Tag Slack-integrated release
git tag -a v1.0-slack-integrated -m "Slack Integration - Feature Testing"
git push origin v1.0-slack-integrated
```

### Docker Image Tags

```bash
# SFDC + PDF tags
rag-app:1.0-sfdc-pdf           # Full version
rag-app:1.0-sfdc-pdf-stable    # Stability marker
rag-app:latest-stable          # Latest stable

# Slack tags
rag-app:1.0-slack-integrated   # Full version
rag-app:1.0-slack-dev          # Development marker
rag-app:latest-development     # Latest development
```

---

## Troubleshooting

### Issue: Port Already in Use

```bash
# Port 5000 in use (SFDC + PDF)
docker ps | grep :5000
docker kill <container-id>

# Port 5001 in use (Slack)
docker ps | grep :5001
docker kill <container-id>

# Or use different ports in docker-compose override
```

### Issue: Database Connection Refused

```bash
# Check PostgreSQL running
docker ps | grep postgres

# Check network
docker network ls | grep rag-network

# Check DNS resolution
docker exec rag-app ping postgres-sfdc-pdf
docker exec rag-app ping postgres-slack-integrated
```

### Issue: Image Not Found

```bash
# Rebuild image
docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .

# Or pull from registry
docker pull your-registry.azurecr.io/rag-app:1.0-sfdc-pdf
```

---

## Summary Table

| Task | SFDC + PDF | Slack | Both |
|------|-----------|-------|------|
| **Build** | `docker build -f Dockerfile.sfdc-pdf` | `docker build -f Dockerfile.slack-integrated` | Build separately |
| **Deploy** | `docker-compose.sfdc-pdf.yml` | `docker-compose.slack-integrated.yml` | Both simultaneously |
| **Port (App)** | 5000 | 5001 | Different ports |
| **Port (DB)** | 5432 | 5433 | Different ports |
| **Rollback** | `git checkout main` | `git checkout feature/slack-integration` | Switch branches |
| **Coexist** | ✅ Yes | ✅ Yes | ✅ Yes |

---

## File Reference

| File | Purpose | Branch |
|------|---------|--------|
| `Dockerfile.sfdc-pdf` | SFDC + PDF image | main |
| `Dockerfile.slack-integrated` | Slack-integrated image | feature/slack-integration |
| `docker-compose.sfdc-pdf.yml` | SFDC + PDF stack | main |
| `docker-compose.slack-integrated.yml` | Slack stack | feature/slack-integration |
| `.env.local` | Local Slack config | feature/slack-integration |
| `.env` | Production config | main |

---

## Next Steps

1. ✅ Build SFDC + PDF image for Monday demo
2. ✅ Build Slack-integrated image for testing
3. ✅ Push both images to registry
4. ✅ Deploy SFDC + PDF for Monday (5000)
5. ✅ Deploy Slack for testing (5001)
6. ✅ Run both simultaneously
7. ✅ Test rollback procedures
8. ✅ Document deployment runbooks for ops team

---

## Support

- **SFDC + PDF**: See main branch documentation
- **Slack**: See feature/slack-integration documentation
- **Switching**: Use quick start commands above
- **Issues**: Check container logs with `docker logs <name>`

