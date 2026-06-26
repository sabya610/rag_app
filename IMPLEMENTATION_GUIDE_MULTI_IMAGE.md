# Multi-Image Deployment Strategy - Complete Implementation Guide

**Status**: ✅ Complete  
**Date**: January 27, 2026  
**Branches**: `main` + `feature/slack-integration`

---

## 🎯 Objective

Create separate Docker images and deployment configurations that:
1. **Protect Monday's SFDC/PDF demo** on main branch
2. **Enable Slack testing** on feature/slack-integration branch
3. **Allow easy switching** between configurations without data loss
4. **Support simultaneous deployment** of both versions if needed
5. **Enable quick rollback** to production-stable SFDC+PDF

---

## 📊 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   RAG APP DEPLOYMENT STRATEGY                   │
└─────────────────────────────────────────────────────────────────┘

                    GitHub Repository
                           │
            ┌──────────────┴──────────────┐
            │                             │
        main branch                  feature/slack-integration
        (Production)                  (Feature Testing)
            │                             │
      ┌─────▼──────┐              ┌─────▼──────────┐
      │ Dockerfile │              │  Dockerfile    │
      │ .sfdc-pdf  │              │  .slack-integ  │
      └─────┬──────┘              └─────┬──────────┘
            │                             │
      ┌─────▼──────────┐          ┌─────▼──────────────┐
      │   Image Tag    │          │   Image Tag        │
      │ 1.0-sfdc-pdf   │          │1.0-slack-integrated│
      │  latest-stable │          │latest-development │
      └─────┬──────────┘          └─────┬──────────────┘
            │                             │
      ┌─────▼────────────┐        ┌─────▼──────────────┐
      │ docker-compose   │        │  docker-compose    │
      │ .sfdc-pdf.yml    │        │ .slack-integ.yml   │
      │ Port: 5000/5432  │        │ Port: 5001/5433    │
      └─────┬────────────┘        └─────┬──────────────┘
            │                             │
      ┌─────▼────────────┐        ┌─────▼──────────────┐
      │  SFDC + PDF RAG  │        │ SFDC + PDF + Slack │
      │  (Monday Demo)   │        │ (Feature Testing)  │
      └──────────────────┘        └────────────────────┘
```

---

## 🗂️ File Structure

### Main Branch Files

```
rag_app/
├── Dockerfile                      (original, kept for compatibility)
├── Dockerfile.sfdc-pdf             ✨ NEW - Production image
├── docker-compose.yml              (original)
├── docker-compose.sfdc-pdf.yml     ✨ NEW - Production stack
├── DEPLOYMENT_STRATEGY.md          ✨ NEW - Architecture overview
├── DEPLOYMENT_REFERENCE_CARD.md    ✨ NEW - Quick ops guide
├── deploy-images.sh                ✨ NEW - Deployment script
├── app/
│   ├── routes/
│   │   └── rag_routes.py          (existing SFDC/PDF routes)
│   └── services/
│       ├── sfdc_client.py         (existing)
│       └── populate_db.py         (existing)
└── ... (all existing files unchanged)
```

### Feature Branch Files

```
rag_app/
├── Dockerfile.slack-integrated     ✨ NEW - Slack feature image
├── docker-compose.slack-integrated.yml  ✨ NEW - Slack test stack
├── MULTI_IMAGE_DEPLOYMENT.md       ✨ NEW - Complete guide
├── SLACK_INTEGRATION.md            (existing)
├── SLACK_INTEGRATION_IMPLEMENTATION.md (existing)
├── SLACK_DEPLOYMENT_GUIDE.md       (existing)
├── app/
│   ├── routes/
│   │   ├── rag_routes.py          (existing)
│   │   └── slack_routes.py        ✨ NEW - Slack endpoints
│   └── services/
│       ├── sfdc_client.py         (existing)
│       ├── slack_client.py        ✨ NEW
│       ├── slack_import.py        ✨ NEW
│       └── populate_db.py         (existing)
└── ... (all other files)
```

---

## 🚀 Deployment Scenarios

### Scenario 1: Monday Morning - SFDC + PDF Demo

**Timeline**: 09:45 - 17:00  
**Branch**: `main`  
**Configuration**: SFDC + PDF only (no Slack)

```bash
# Friday afternoon - Build image
git checkout main
docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .
docker push your-registry.azurecr.io/rag-app:1.0-sfdc-pdf

# Monday morning - Deploy
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d

# Verify running
curl http://localhost:5000/api/rag/stats

# Result:
# ✅ SFDC integration working
# ✅ PDF knowledge base loaded
# ✅ LLM responding
# ✅ UI accessible at http://localhost:5000
```

**Safety**: No Slack code is even loaded. Complete isolation.

---

### Scenario 2: Monday Evening - Switch to Slack Testing

**Timeline**: 17:00 - ongoing  
**Branch**: `feature/slack-integration`  
**Configuration**: SFDC + PDF + Slack

```bash
# Stop demo
docker-compose -f docker-compose.sfdc-pdf.yml down

# Switch to feature branch
git checkout feature/slack-integration

# Build new image (or pull from registry)
docker build -f Dockerfile.slack-integrated -t rag-app:1.0-slack-integrated .

# Deploy on different port (5001 instead of 5000)
docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d

# Initialize Slack database
docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py

# Verify
curl http://localhost:5001/api/slack/stats

# Result:
# ✅ Separate database (no SFDC data affected)
# ✅ Slack import working
# ✅ Message search available
# ✅ Testing isolated from production
```

---

### Scenario 3: Parallel Testing - Both Running

**Timeline**: Tuesday+  
**Branches**: Both main and feature/slack-integration  
**Configuration**: Side-by-side comparison

```bash
# Terminal 1: Start SFDC + PDF
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up

# Terminal 2: Start Slack (in different directory or with -p prefix)
git checkout feature/slack-integration
docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up

# Both running simultaneously:
# - SFDC + PDF: http://localhost:5000
# - Slack: http://localhost:5001
# - Separate databases: 5432 vs 5433
# - Full side-by-side comparison possible
```

---

### Scenario 4: Emergency Rollback - Revert to Production

**Scenario**: Issues with Slack integration testing  
**Action**: One-command rollback

```bash
# Stop everything
docker-compose -f docker-compose.slack-integrated.yml down

# Revert to main and production image
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d

# Back to production in ~30 seconds
curl http://localhost:5000/api/rag/stats

# SFDC + PDF demo running on production configuration
```

---

## 📦 Image Details

### SFDC + PDF Image (Production)

**File**: `Dockerfile.sfdc-pdf`  
**Tag**: `rag-app:1.0-sfdc-pdf`  
**Base**: `python:3.10-slim`  
**Features**:
- Python dependencies (no Slack SDK)
- SFDC integration enabled
- PDF processing
- Llama.cpp model
- Sentence-transformers for embeddings
- Health checks for `/api/rag/stats`

**Environment**:
```
FEATURES_SLACK=disabled
```

**Size**: ~3.5 GB (with models)

---

### Slack-Integrated Image (Feature)

**File**: `Dockerfile.slack-integrated`  
**Tag**: `rag-app:1.0-slack-integrated`  
**Base**: `python:3.10-slim`  
**Features**:
- All SFDC + PDF features
- Slack SDK included
- Slack import service
- Thread management
- pgvector semantic search
- Health checks for `/api/slack/stats`

**Environment**:
```
FEATURES_SLACK=enabled
```

**Size**: ~3.7 GB (with Slack dependencies + models)

---

## 🔄 Docker Compose Configuration

### SFDC + PDF Stack (docker-compose.sfdc-pdf.yml)

```yaml
Services:
  - postgres-sfdc-pdf:5432 (PostgreSQL with pgvector)
  - rag-app-sfdc-pdf:5000 (Flask app with SFDC)

Networks:
  - rag-network (isolated network)

Volumes:
  - postgres_sfdc_pdf_data (production data)
  - rag_app_sfdc_pdf_logs (app logs)

Environment:
  - SFDC_* (all SFDC credentials)
  - DB_HOST: postgres-sfdc-pdf
  - FEATURES_SLACK: disabled
```

### Slack-Integrated Stack (docker-compose.slack-integrated.yml)

```yaml
Services:
  - postgres-slack-integrated:5433 (PostgreSQL with pgvector)
  - rag-app-slack-integrated:5001 (Flask app with Slack)

Networks:
  - rag-network-slack (isolated network, different from SFDC)

Volumes:
  - postgres_slack_integrated_data (test data)
  - rag_app_slack_integrated_logs (app logs)

Environment:
  - SFDC_* (all SFDC credentials)
  - SLACK_* (all Slack credentials)
  - DB_HOST: postgres-slack-integrated
  - FEATURES_SLACK: enabled
```

---

## 🎛️ Switching Between Images

### Step 1: Stop Current Stack

```bash
# Stop current deployment
docker-compose -f docker-compose.sfdc-pdf.yml down
# OR
docker-compose -f docker-compose.slack-integrated.yml down
```

### Step 2: Switch Branch

```bash
# Switch to desired branch
git checkout main                        # For SFDC + PDF
# OR
git checkout feature/slack-integration   # For Slack
```

### Step 3: Deploy New Stack

```bash
# Deploy new configuration
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
# OR
docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d
```

### Step 4: Verify

```bash
# Check status
docker ps | grep rag-app

# Test health endpoint
curl http://localhost:5000/api/rag/stats        # SFDC
# OR
curl http://localhost:5001/api/slack/stats      # Slack
```

---

## 📋 Checklists

### Pre-Monday Deployment

- [ ] SFDC credentials configured in `.env.demo-sfdc`
- [ ] Dockerfile.sfdc-pdf tested locally
- [ ] docker-compose.sfdc-pdf.yml validated
- [ ] Image built and pushed to registry
- [ ] Health check working
- [ ] SFDC connection verified
- [ ] PDF knowledge base loaded
- [ ] All search types working

### Pre-Slack Testing

- [ ] Switch to feature/slack-integration branch
- [ ] Slack app created at https://api.slack.com/apps
- [ ] Slack bot token and signing secret obtained
- [ ] `.env.test-slack` configured with credentials
- [ ] Dockerfile.slack-integrated built locally
- [ ] docker-compose.slack-integrated.yml tested
- [ ] Database initialized with `init_slack_db.py`
- [ ] Message import tested
- [ ] Search endpoints responding

### Rollback Verification

- [ ] Can switch branches without errors
- [ ] Can stop Slack stack and start SFDC stack
- [ ] SFDC data persists across restarts
- [ ] No port conflicts when switching
- [ ] Health checks pass after rollback

---

## 🔐 Environment Variables

### .env.demo-sfdc (Main Branch)

```bash
DB_USER=postgres
DB_PASS=postgres
DB_NAME=ragdb
MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
EMBEDDING_MODEL=./models/embedding/all-MiniLM-L6-v2
SFDC_USERNAME=your-email@company.com
SFDC_PASSWORD=your_password
SFDC_SECURITY_TOKEN=token123
SFDC_CLIENT_ID=client_id
SFDC_CLIENT_SECRET=client_secret
```

### .env.test-slack (Feature Branch)

```bash
DB_USER=postgres
DB_PASS=postgres
DB_NAME=ragdb
MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
EMBEDDING_MODEL=./models/embedding/all-MiniLM-L6-v2
# SFDC
SFDC_USERNAME=your-email@company.com
SFDC_PASSWORD=your_password
SFDC_SECURITY_TOKEN=token123
SFDC_CLIENT_ID=client_id
SFDC_CLIENT_SECRET=client_secret
# Slack
SLACK_BOT_TOKEN=xoxb-YOUR-TOKEN
SLACK_SIGNING_SECRET=YOUR-SECRET
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30
```

---

## 🚀 Deployment Script

Use `deploy-images.sh` for automated operations:

```bash
# Main branch - Production operations
git checkout main

# Build production image
./deploy-images.sh build

# Deploy production stack
./deploy-images.sh deploy

# Check status
./deploy-images.sh status

# Run health checks
./deploy-images.sh test

# Push to registry
./deploy-images.sh push your-registry.azurecr.io

# View logs
./deploy-images.sh logs
```

---

## 📊 Testing Matrix

| Test | SFDC+PDF | Slack | Both |
|------|----------|-------|------|
| **Build image** | ✅ | ✅ | Sequential |
| **Deploy stack** | ✅ | ✅ | Parallel |
| **Health check** | Port 5000 | Port 5001 | Both |
| **Database** | 5432 | 5433 | Isolated |
| **Data persistence** | ✅ | ✅ | Separate |
| **Rollback** | Instant | Instant | Supported |
| **Switch branches** | ✅ | ✅ | Both ways |

---

## 🔗 Kubernetes Support

### Deploy SFDC + PDF to K8s

```bash
# Update helm/values.yaml
# Then deploy
helm upgrade --install rag-app-demo helm/ \
  -f helm/values-sfdc-pdf.yaml \
  -n rag-app-demo
```

### Deploy Slack to K8s

```bash
# Update helm/values.yaml with Slack config
# Then deploy separately
helm upgrade --install rag-app-slack helm/ \
  -f helm/values-slack-integrated.yaml \
  -n rag-app-slack
```

### Both Running in K8s

```bash
# Deploy both in separate namespaces
helm upgrade --install rag-app-demo helm/ \
  -f helm/values-sfdc-pdf.yaml \
  -n rag-app-demo --create-namespace

helm upgrade --install rag-app-slack helm/ \
  -f helm/values-slack-integrated.yaml \
  -n rag-app-slack --create-namespace
```

---

## 📞 Troubleshooting

### Issue: Port Already in Use

```bash
# Find and kill existing process
docker ps | grep :5000 | awk '{print $1}' | xargs docker kill
docker ps | grep :5432 | awk '{print $1}' | xargs docker kill
```

### Issue: Database Connection Failed

```bash
# Check PostgreSQL health
docker logs postgres-sfdc-pdf      # SFDC database
docker logs postgres-slack-integrated # Slack database

# Verify network connectivity
docker network inspect rag-network
docker network inspect rag-network-slack
```

### Issue: Can't Switch Branches

```bash
# Ensure no uncommitted changes
git status

# Stash any uncommitted work
git stash

# Try switch again
git checkout main
```

### Issue: Image Not Found

```bash
# Rebuild locally
git checkout main
docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .

# Or pull from registry
docker pull your-registry.azurecr.io/rag-app:1.0-sfdc-pdf
```

---

## 📈 Scaling Considerations

### Horizontal Scaling

```yaml
# Kubernetes HPA for SFDC + PDF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-app-demo-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-app-demo
  minReplicas: 2
  maxReplicas: 5
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
# Nginx reverse proxy
upstream rag_app_demo {
    server localhost:5000;
}

upstream rag_app_slack {
    server localhost:5001;
}

server {
    listen 80;
    
    location /demo/ {
        proxy_pass http://rag_app_demo/;
    }
    
    location /slack/ {
        proxy_pass http://rag_app_slack/;
    }
}
```

---

## ✅ Summary

### What We Achieved

✅ **Separate stable image** for Monday SFDC/PDF demo  
✅ **Separate feature image** for Slack integration  
✅ **Easy switching** between configurations  
✅ **Parallel deployment** support (both running)  
✅ **Quick rollback** capability (30 seconds)  
✅ **Data isolation** (separate databases)  
✅ **Production-ready** infrastructure  
✅ **Kubernetes support** for enterprise deployment  

### Key Benefits

1. **Monday Demo is Protected**: Uses main branch, only SFDC/PDF code
2. **Feature Development is Isolated**: Uses feature branch, includes Slack
3. **Easy to Switch**: One branch checkout, one docker-compose command
4. **Quick Rollback**: Stop Slack, start SFDC in 30 seconds
5. **Both Can Run**: Full comparison testing if needed
6. **Scalable**: Ready for Kubernetes production deployment

---

## 📞 Next Steps

1. ✅ **This week**: Test both configurations locally
2. ✅ **Friday PM**: Build and push both images to registry
3. ✅ **Monday 09:45**: Deploy SFDC + PDF demo
4. ✅ **Monday 17:00**: Deploy Slack integration for testing
5. ✅ **Tuesday+**: Performance testing and optimization
6. ✅ **Week 2**: Merge Slack to main after successful testing

---

**Date**: January 27, 2026  
**Version**: 1.0  
**Status**: ✅ Complete & Ready for Deployment

