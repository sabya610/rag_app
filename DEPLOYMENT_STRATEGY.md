# Production Deployment Strategy

**Purpose**: Multi-environment deployment with easy switching between stable SFDC/PDF demo and new Slack integration testing environment.

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│         PRODUCTION DEPLOYMENT STRATEGY          │
└─────────────────────────────────────────────────┘

Main Branch (Stable)              Feature Branch (Development)
    ↓                                    ↓
rag-app:1.0-sfdc-pdf          rag-app:1.0-slack-integrated
docker-compose.sfdc-pdf.yml   docker-compose.slack-integrated.yml
localhost:5000                 localhost:5001
localhost:5432                 localhost:5433
SFDC + PDF                     SFDC + PDF + Slack
```

---

## Quick Reference

| Aspect | SFDC + PDF (Production) | Slack (Testing) |
|--------|------------------------|-----------------| 
| Branch | `main` | `feature/slack-integration` |
| Image | `rag-app:1.0-sfdc-pdf` | `rag-app:1.0-slack-integrated` |
| Port | 5000 | 5001 |
| DB Port | 5432 | 5433 |
| Status | ✅ Production Ready | 🚀 Feature Testing |

---

## Deployment Scenarios

### Scenario 1: Monday Demo (SFDC + PDF)

```bash
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
curl http://localhost:5000/api/rag/stats
```

### Scenario 2: Post-Demo Testing (Slack Integration)

```bash
git checkout feature/slack-integration
docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d
curl http://localhost:5001/api/slack/stats
```

### Scenario 3: Both Running Simultaneously

```bash
# Terminal 1: Main branch - SFDC + PDF
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml up -d

# Terminal 2: Feature branch - Slack
git checkout feature/slack-integration
docker-compose -f docker-compose.slack-integrated.yml up -d
```

---

## Key Files

| File | Purpose | Branch |
|------|---------|--------|
| `Dockerfile.sfdc-pdf` | Stable production image | main |
| `docker-compose.sfdc-pdf.yml` | SFDC production stack | main |
| `DEPLOYMENT_REFERENCE_CARD.md` | Quick ops reference | both |
| `deploy-images.sh` | Automated deployment script | both |
| `Dockerfile.slack-integrated` | Slack feature image | feature/slack-integration |
| `docker-compose.slack-integrated.yml` | Slack test stack | feature/slack-integration |
| `MULTI_IMAGE_DEPLOYMENT.md` | Complete deployment guide | feature/slack-integration |

---

## See Also

For detailed deployment instructions:
- **DEPLOYMENT_REFERENCE_CARD.md** - Quick commands for operators
- **deploy-images.sh** - Automated build/push/deploy script

For Slack integration specific deployment:
- **feature/slack-integration branch → MULTI_IMAGE_DEPLOYMENT.md**

---

