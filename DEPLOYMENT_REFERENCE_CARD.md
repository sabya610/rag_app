# Quick Deployment Reference Card

**For SFDC + PDF (Main Branch - Production)**

---

## ⚡ Deploy Production (SFDC + PDF)

```bash
cd /opt/rag_app
git checkout main
docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
sleep 10 && curl http://localhost:5000/api/rag/stats
```

**Access**: http://localhost:5000

---

## 🔍 Health Check

```bash
# Container running?
docker ps | grep rag-app-sfdc-pdf

# App responsive?
curl http://localhost:5000/api/rag/stats

# Logs?
docker logs rag-app-sfdc-pdf | tail -20
```

---

## 🐛 Troubleshooting

### App won't start
```bash
# Check logs
docker logs rag-app-sfdc-pdf

# Restart
docker restart rag-app-sfdc-pdf
```

### Port already in use
```bash
# Kill existing container
docker ps | grep :5000 | awk '{print $1}' | xargs docker kill
```

### Database issues
```bash
# Backup data
docker exec postgres-sfdc-pdf pg_dump -U postgres ragdb > backup.sql

# Reset
docker-compose -f docker-compose.sfdc-pdf.yml down -v
docker-compose -f docker-compose.sfdc-pdf.yml up -d
```

---

## 📊 Monitoring

```bash
# Resource usage
docker stats rag-app-sfdc-pdf

# Container size
docker ps --format "table {{.Names}}\t{{.Size}}" | grep rag-app

# Check image
docker images | grep rag-app
```

---

## 🚀 Push to Registry

```bash
# Tag image
docker tag rag-app:1.0-sfdc-pdf your-registry.azurecr.io/rag-app:1.0-sfdc-pdf

# Push to Azure ACR
docker push your-registry.azurecr.io/rag-app:1.0-sfdc-pdf
```

---

## 📝 Environment Setup

```bash
cat > /opt/rag_app/.env.demo-sfdc << 'EOF'
DB_USER=postgres
DB_PASS=postgres
DB_NAME=ragdb
SFDC_USERNAME=your-sfdc-email@company.com
SFDC_PASSWORD=YOUR_PASSWORD_HERE
SFDC_SECURITY_TOKEN=YOUR_TOKEN_HERE
SFDC_CLIENT_ID=YOUR_CLIENT_ID_HERE
SFDC_CLIENT_SECRET=YOUR_CLIENT_SECRET_HERE
EOF
```

---

## 🔗 Related Documentation

- **Full deployment guide**: See `DEPLOYMENT_STRATEGY.md` in main branch
- **Multi-image setup**: Switch to `feature/slack-integration` branch for `MULTI_IMAGE_DEPLOYMENT.md`
- **Slack integration**: See `feature/slack-integration` branch

---

**Last Updated**: January 27, 2026

