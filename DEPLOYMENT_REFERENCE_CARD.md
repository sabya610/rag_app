# Quick Deployment Reference Card

**Keep this handy for fast image switching**

---

## 📌 Current Configuration

| Property | SFDC + PDF | Slack |
|----------|-----------|-------|
| **Status** | ✅ Production | 🚀 Testing |
| **Port** | 5000 | 5001 |
| **DB Port** | 5432 | 5433 |
| **Container** | rag-app-sfdc-pdf | rag-app-slack-integrated |
| **Image** | rag-app:1.0-sfdc-pdf | rag-app:1.0-slack-integrated |
| **Branch** | `main` | `feature/slack-integration` |

---

## ⚡ Quick Commands

### Deploy SFDC + PDF (Monday Demo)

```bash
cd /opt/rag_app
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
sleep 5 && curl http://localhost:5000/api/rag/stats
```

**Access**: http://localhost:5000

---

### Deploy Slack Integration

```bash
cd /opt/rag_app
git checkout feature/slack-integration
docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d
sleep 5 && docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py
sleep 5 && curl http://localhost:5001/api/slack/stats
```

**Access**: http://localhost:5001

---

### Run Both Simultaneously

```bash
# Terminal 1
cd /opt/rag_app && git checkout main && \
docker-compose -f docker-compose.sfdc-pdf.yml up -d

# Terminal 2
cd /opt/rag_app && git checkout feature/slack-integration && \
docker-compose -f docker-compose.slack-integrated.yml up -d
```

**Access**: 
- SFDC + PDF: http://localhost:5000
- Slack: http://localhost:5001

---

### Emergency Rollback (SFDC + PDF)

```bash
docker-compose -f docker-compose.slack-integrated.yml down
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml up -d
curl http://localhost:5000/api/rag/stats
```

**Time to rollback**: ~30 seconds

---

## 🔍 Health Checks

### SFDC + PDF

```bash
# Container running?
docker ps | grep rag-app-sfdc-pdf

# App responsive?
curl http://localhost:5000/api/rag/stats

# Database connected?
docker exec postgres-sfdc-pdf pg_isready
```

### Slack Integration

```bash
# Container running?
docker ps | grep rag-app-slack-integrated

# App responsive?
curl http://localhost:5001/api/slack/stats

# Database connected?
docker exec postgres-slack-integrated pg_isready
```

---

## 🐛 Troubleshooting

### App won't start

```bash
# Check logs
docker logs rag-app-sfdc-pdf        # SFDC
docker logs rag-app-slack-integrated # Slack

# Check database
docker logs postgres-sfdc-pdf
docker logs postgres-slack-integrated

# Restart container
docker restart rag-app-sfdc-pdf
docker restart rag-app-slack-integrated
```

### Port conflicts

```bash
# Check what's using port 5000
lsof -i :5000

# Kill container
docker ps | grep :5000 | awk '{print $1}' | xargs docker kill
```

### Database issues

```bash
# Backup current data
docker exec postgres-sfdc-pdf pg_dump -U postgres ragdb > backup.sql

# Reset database
docker-compose -f docker-compose.sfdc-pdf.yml down -v  # Remove volumes!
docker-compose -f docker-compose.sfdc-pdf.yml up -d    # Fresh start
```

---

## 📊 Monitoring

### Disk Usage

```bash
# Docker images
docker images | grep rag-app

# Docker volumes
docker volume ls | grep rag

# Container disk
docker ps --format "table {{.Names}}\t{{.Size}}"
```

### Memory Usage

```bash
# Real-time
docker stats rag-app-sfdc-pdf
docker stats rag-app-slack-integrated

# Limits
docker inspect rag-app-sfdc-pdf --format='{{.HostConfig.Memory}}'
```

### Network

```bash
# Check network bridge
docker network inspect rag-network
docker network inspect rag-network-slack

# Check container IPs
docker inspect -f '{{.Name}} - {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
  rag-app-sfdc-pdf rag-app-slack-integrated
```

---

## 🚀 Deployment Timeline (Monday)

| Time | Task | Command |
|------|------|---------|
| **09:45** | Check SFDC + PDF ready | `curl http://localhost:5000/api/rag/stats` |
| **09:50** | Final health check | `docker ps \| grep sfdc-pdf` |
| **10:00** | Start demo | App already running |
| **17:00** | Stop demo | `docker-compose -f docker-compose.sfdc-pdf.yml down` |
| **17:05** | Deploy Slack testing | `git checkout feature/slack-integration` |
| **17:10** | Start Slack stack | `docker-compose -f docker-compose.slack-integrated.yml up -d` |

---

## 📝 Post-Deployment Checklist

### After SFDC + PDF Deploy

- [ ] `curl http://localhost:5000/api/rag/stats` returns 200
- [ ] SFDC articles loaded
- [ ] PDF knowledge base loaded
- [ ] Search working correctly
- [ ] LLM responses accurate

### After Slack Deploy

- [ ] `curl http://localhost:5001/api/slack/stats` returns 200
- [ ] Database initialized
- [ ] Slack connection verified
- [ ] Message import working
- [ ] Search endpoints responding

---

## 🔐 Environment Files

### Create for SFDC + PDF

```bash
cat > /opt/rag_app/.env.demo-sfdc << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_NAME=ragdb
SFDC_USERNAME=demo@company.com
SFDC_PASSWORD=password123
SFDC_SECURITY_TOKEN=token123
SFDC_CLIENT_ID=id123
SFDC_CLIENT_SECRET=secret123
EOF
```

### Create for Slack Testing

```bash
cat > /opt/rag_app/.env.test-slack << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_NAME=ragdb
SFDC_USERNAME=demo@company.com
SFDC_PASSWORD=YOUR_SFDC_PASSWORD_HERE
SFDC_SECURITY_TOKEN=YOUR_SFDC_TOKEN_HERE
SFDC_CLIENT_ID=YOUR_CLIENT_ID_HERE
SFDC_CLIENT_SECRET=YOUR_CLIENT_SECRET_HERE
SLACK_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
SLACK_SIGNING_SECRET=YOUR_SIGNING_SECRET_HERE
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30
EOF
```

---

## 📞 Emergency Contact

**Issue**: App not responding  
**Action**: Check logs → Restart container → Rollback if needed

**Issue**: Database corrupted  
**Action**: Restore from backup → Restart services

**Issue**: Total failure  
**Action**: `docker-compose down -v` → Rebuild image → Deploy fresh

---

## 🔗 Useful Links

- **GitHub SFDC + PDF**: https://github.com/sabya610/rag_app (main branch)
- **GitHub Slack**: https://github.com/sabya610/rag_app/tree/feature/slack-integration
- **Docker Images**: View with `docker images | grep rag-app`
- **Container Logs**: `docker logs <container-name>`
- **Database Access**: `docker exec -it postgres-sfdc-pdf psql -U postgres`

---

## 📋 Deployment Runbook

### Pre-Deployment (Friday)

```bash
# 1. Build images
git checkout main && docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .
git checkout feature/slack-integration && docker build -f Dockerfile.slack-integrated -t rag-app:1.0-slack-integrated .

# 2. Test SFDC + PDF
docker-compose -f docker-compose.sfdc-pdf.yml up -d
sleep 30 && curl http://localhost:5000/api/rag/stats
docker-compose -f docker-compose.sfdc-pdf.yml down

# 3. Test Slack
docker-compose -f docker-compose.slack-integrated.yml up -d
sleep 30 && curl http://localhost:5001/api/slack/stats
docker-compose -f docker-compose.slack-integrated.yml down

# ✅ Both ready for Monday
```

### Monday Morning (09:45 AM)

```bash
# 1. Deploy SFDC + PDF
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d

# 2. Wait for startup
sleep 30

# 3. Verify running
curl http://localhost:5000/api/rag/stats

# ✅ Ready for demo
```

### Monday Evening (17:00)

```bash
# 1. Stop SFDC + PDF
docker-compose -f docker-compose.sfdc-pdf.yml down

# 2. Switch to Slack
git checkout feature/slack-integration

# 3. Deploy Slack
docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d

# 4. Initialize DB
sleep 30
docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py

# 5. Verify running
curl http://localhost:5001/api/slack/stats

# ✅ Testing environment ready
```

---

**Last Updated**: January 27, 2026  
**Valid For**: Deployment cycle through Q1 2026

