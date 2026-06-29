# RAG App Slack Integration - Helm Package & Deployment Summary

**Date:** 2026-06-26  
**Status:** ✅ **READY FOR DEPLOYMENT**  
**Repository:** https://github.com/sabya610/rag_app  
**Branches:**
  - `main`: Production stable (SFDC+PDF only)
  - `feature/slack-integration`: Slack integration feature branch

---

## 📦 What You Have Now

### 1. **Complete Helm Chart** (`helm/rag-app-slack/`)
A production-ready Kubernetes Helm chart for deploying RAG App with Slack integration on your FTC AIE 1.1.1 cluster.

**Chart Components:**
- ✅ Deployment (with 2 replicas by default)
- ✅ Service (ClusterIP for internal access)
- ✅ ConfigMap (non-sensitive configuration)
- ✅ Secrets (Slack, SFDC credentials)
- ✅ ServiceAccount (RBAC support)
- ✅ HorizontalPodAutoscaler (auto-scaling 2-4 pods)
- ✅ PostgreSQL database (via Bitnami dependency)
- ✅ Health checks (liveness & readiness probes)

**Version:** 1.0.0  
**Chart Location:** `helm/rag-app-slack/`

### 2. **Automated Deployment Scripts**

#### **PowerShell (Windows):**
- `helm/Deploy-Helm.ps1` - Main deployment script
- `helm/Package-Helm.ps1` - Package for distribution
- **Usage:** `.\helm\Deploy-Helm.ps1 -Action deploy`

#### **Bash (Linux/Mac/WSL):**
- `helm/deploy-helm.sh` - Main deployment script  
- `helm/package-helm.sh` - Package for distribution
- **Usage:** `chmod +x helm/deploy-helm.sh && ./helm/deploy-helm.sh deploy`

### 3. **Cluster-Specific Configuration**
- `helm/values-ftc-aie.yaml` - FTC AIE 1.1.1 optimized values
  - 2 replicas, 3-6GB memory/1.5-3 CPU
  - 100GB PostgreSQL storage
  - Auto-scaling enabled (min 2, max 4 pods)

### 4. **Comprehensive Documentation**

| Document | Purpose | Pages |
|----------|---------|-------|
| `helm/HELM_DEPLOYMENT_GUIDE.md` | Complete deployment guide | 20+ |
| `helm/QUICK_REFERENCE.md` | Quick commands & troubleshooting | 2 |
| `DEPLOYMENT_CHECKLIST.md` | Pre-deployment verification | 10+ |
| GitHub Repo Wiki | API documentation | - |

### 5. **Alternative: Docker Compose for VMs**
If you don't want to use Kubernetes:
- `Deploy-ToCluster.ps1` - Deploy to VM cluster via SSH
- `deploy-to-cluster.sh` - Bash version
- Uses Docker Compose instead of Helm
- Separate databases prevent data mixing

---

## 🚀 Quick Start (3 Steps)

### **Step 1: Get Slack Credentials**

```
Go to: https://api.slack.com/apps
1. Create app or select existing
2. Get Bot Token (starts with xoxb-)
3. Get Signing Secret (32+ chars)
4. Verify bot has channel access
```

### **Step 2: Ensure Cluster Access**

```powershell
# Windows: Test kubectl connectivity
kubectl cluster-info
kubectl get nodes
```

### **Step 3: Run Deployment**

```powershell
# Windows PowerShell
cd c:\Users\malliks\rag_app
.\helm\Deploy-Helm.ps1 -Action deploy

# When prompted:
# - Paste Slack Bot Token
# - Paste Slack Signing Secret
# - Enter SFDC credentials (optional)
# - Confirm deployment
```

```bash
# Linux/Mac/WSL
cd /path/to/rag_app
chmod +x helm/deploy-helm.sh
./helm/deploy-helm.sh deploy
```

---

## 📥 Download & Installation

### Option 1: Git Clone (Recommended)

```bash
git clone https://github.com/sabya610/rag_app.git
cd rag_app
git checkout main  # For production
# or
git checkout feature/slack-integration  # For feature testing
```

### Option 2: Create Download Package

Create a distribution-ready package:

```powershell
# Windows
.\helm\Package-Helm.ps1 -Action package
# Creates: helm-releases/rag-app-slack-helm-*.zip
```

```bash
# Linux/Mac
chmod +x helm/package-helm.sh
./helm/package-helm.sh
# Creates: helm-releases/rag-app-slack-helm-*.tar.gz
```

### Option 3: Download Individual Files

Download from GitHub:
- Helm Chart: `helm/rag-app-slack/`
- Deployment Script: `helm/Deploy-Helm.ps1` or `helm/deploy-helm.sh`
- Values: `helm/values-ftc-aie.yaml`
- Guide: `helm/HELM_DEPLOYMENT_GUIDE.md`

---

## 📋 Deployment Options

### **Option A: Kubernetes with Helm (RECOMMENDED)**

**Best for:** Production clusters, FTC AIE 1.1.1  
**Deployment time:** ~3-5 minutes  
**Rollback:** ✅ Easy (helm rollback)  
**Scaling:** ✅ Automatic (HPA enabled)

```bash
helm install rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set slack.botToken=xoxb-YOUR_TOKEN \
  --set slack.signingSecret=YOUR_SECRET
```

### **Option B: Docker Compose on VMs**

**Best for:** Traditional VM clusters, easier management  
**Deployment time:** ~5-10 minutes  
**Rollback:** ✅ Manual (stop/git checkout/restart)  
**Scaling:** 🔄 Manual scaling needed

```bash
ssh root@10.227.81.151
cd /opt/rag_app
docker-compose -f docker-compose.slack-integrated.yml up -d
```

### **Option C: Manual kubectl Commands**

**Best for:** Learning/debugging  
**Deployment time:** ~10-15 minutes  
**Learning curve:** Steep

```bash
kubectl apply -f <generated-yaml-files>
```

---

## ✅ Verification Checklist

After deployment, verify everything works:

```bash
# 1. Check deployment status
kubectl get deployment rag-app-slack -n default

# 2. Check pods are running
kubectl get pods -n default -l app.kubernetes.io/instance=rag-app-slack

# 3. View logs
kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack -f

# 4. Port-forward for testing
kubectl port-forward -n default svc/rag-app-slack 5001:5001

# 5. Test health endpoint
curl http://localhost:5001/api/slack/stats

# 6. Test Slack channels
curl http://localhost:5001/api/slack/channels

# 7. Test Slack import
curl -X POST http://localhost:5001/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{"channel_ids":["C123456"],"days_back":7}'
```

---

## 🗂️ File Structure

```
rag_app/
├── helm/
│   ├── rag-app-slack/                    # Main Helm chart
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/
│   │       ├── deployment.yaml           # K8s deployment
│   │       ├── service.yaml              # K8s service
│   │       ├── configmap.yaml            # Configuration
│   │       ├── secret.yaml               # Secrets
│   │       ├── serviceaccount.yaml
│   │       ├── hpa.yaml                  # Auto-scaling
│   │       ├── _helpers.tpl
│   │       └── NOTES.txt                 # Post-install notes
│   ├── values-ftc-aie.yaml               # FTC cluster values
│   ├── Deploy-Helm.ps1                   # PowerShell deployment
│   ├── deploy-helm.sh                    # Bash deployment
│   ├── Package-Helm.ps1                  # Package creation
│   ├── package-helm.sh                   # Package creation
│   ├── HELM_DEPLOYMENT_GUIDE.md          # Full guide (20+ pages)
│   └── QUICK_REFERENCE.md                # Quick commands
├── Deploy-ToCluster.ps1                  # VM deployment (PowerShell)
├── deploy-to-cluster.sh                  # VM deployment (Bash)
├── DEPLOYMENT_CHECKLIST.md               # Pre-deployment checklist
├── requirements.txt                      # Python dependencies
├── Dockerfile.slack-integrated           # Docker image
├── docker-compose.slack-integrated.yml   # Compose file
├── app/
│   ├── models.py                         # DB models (SlackMessage, SlackThread)
│   ├── config.py                         # Configuration
│   ├── services/
│   │   ├── slack_client.py              # Slack API wrapper
│   │   └── slack_import.py              # Message import service
│   └── routes/
│       └── slack_routes.py               # API endpoints
└── init_slack_db.py                      # Database initialization
```

---

## 🔧 Key Configuration Values

### Helm Chart Customization

```yaml
# Default replicas
ragApp.replicaCount: 2

# Resource limits
ragApp.resources.requests.memory: "3Gi"
ragApp.resources.limits.memory: "6Gi"

# Auto-scaling
ragApp.autoscaling.enabled: true
ragApp.autoscaling.minReplicas: 2
ragApp.autoscaling.maxReplicas: 4

# Database size
postgresql.primary.persistence.size: "100Gi"

# Slack settings
slack.importLimit: 100        # Max messages per import
slack.importDays: 30          # Days back to import
```

### Environment Variables

Set at deployment time:

```bash
--set slack.botToken=xoxb-YOUR_TOKEN
--set slack.signingSecret=YOUR_SECRET
--set salesforce.username=your_sfdc_username
--set salesforce.password=your_sfdc_password
--set salesforce.securityToken=your_security_token
```

---

## 📚 API Endpoints

After deployment, these endpoints are available:

```bash
# Get stats/health
GET http://localhost:5001/api/slack/stats

# List channels
GET http://localhost:5001/api/slack/channels

# Import messages from Slack
POST http://localhost:5001/api/slack/import
Body: {"channel_ids":["C123456"],"days_back":30}

# Search messages
POST http://localhost:5001/api/slack/search
Body: {"query":"topic","search_type":"semantic","limit":10}

# Get thread
GET http://localhost:5001/api/slack/threads/{thread_id}
```

---

## 🔄 Update/Rollback

### Update Deployment

```bash
helm upgrade rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set slack.botToken=xoxb-new-token
```

### View History

```bash
helm history rag-app-slack -n default
```

### Rollback to Previous

```bash
helm rollback rag-app-slack 0 -n default
```

---

## 🐛 Troubleshooting

### Pods in CrashLoopBackOff

```bash
# Check logs
kubectl logs -n default POD_NAME -f

# Common cause: Invalid Slack token
# Fix: Update secret and restart pods
kubectl delete pod -n default -l app.kubernetes.io/instance=rag-app-slack
```

### Cannot Connect to Database

```bash
# Check PostgreSQL
kubectl get statefulset -n default

# Check database logs
kubectl logs -n default postgres-slack-0

# Restart database
kubectl delete pod -n default postgres-slack-0
```

### Out of Storage

```bash
# Check PVC
kubectl get pvc -n default

# Increase size
kubectl patch pvc postgres-slack-postgresql -n default \
  -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

See `helm/HELM_DEPLOYMENT_GUIDE.md` for detailed troubleshooting.

---

## 📞 Support

### Documentation

- **Full Guide:** `helm/HELM_DEPLOYMENT_GUIDE.md` (20+ pages)
- **Quick Reference:** `helm/QUICK_REFERENCE.md`
- **Pre-deployment:** `DEPLOYMENT_CHECKLIST.md`
- **API Docs:** Check app/routes/slack_routes.py

### GitHub

- **Repository:** https://github.com/sabya610/rag_app
- **Issues:** Create an issue for problems
- **Slack Integration Branch:** `feature/slack-integration`

### Resources

- **Helm Docs:** https://helm.sh/docs/
- **Kubernetes Docs:** https://kubernetes.io/docs/
- **Slack API:** https://api.slack.com/docs
- **PostgreSQL:** https://www.postgresql.org/docs/

---

## 📦 What's in Each Deployment Package

### Helm Chart Package (`rag-app-slack-*.tgz`)
- Complete Helm chart with all templates
- Default and FTC-specific values
- Ready to deploy to any Kubernetes cluster
- Includes health checks and auto-scaling

### Deployment Bundle (`rag-app-slack-deployment-*.tar.gz`)
- Helm chart (packaged)
- Deployment automation scripts
- Complete documentation
- Ready for offline distribution

---

## 🎯 Next Steps

1. **Get Slack Credentials** (5 min)
   - Go to https://api.slack.com/apps
   - Create/select app
   - Get Bot Token and Signing Secret

2. **Verify Cluster Access** (2 min)
   - `kubectl cluster-info`
   - `kubectl get nodes`

3. **Run Deployment** (3-5 min)
   - `.\helm\Deploy-Helm.ps1 -Action deploy` (Windows)
   - or `./helm/deploy-helm.sh deploy` (Linux/Mac)

4. **Verify Deployment** (2 min)
   - Check pods: `kubectl get pods -n default`
   - Check logs: `kubectl logs -n default ...`
   - Test endpoint: `curl http://localhost:5001/api/slack/stats`

5. **Test Slack Integration** (5-10 min)
   - Import messages
   - Search Slack data
   - Retrieve threads

---

## ⚡ Key Features

✅ **Production Ready**
- Multi-replica deployment
- Auto-scaling support
- Health checks
- Resource limits

✅ **Easy Deployment**
- Automated PowerShell/Bash scripts
- Pre-configured for FTC cluster
- One-command deployment

✅ **Complete Documentation**
- 20+ pages of guides
- Quick reference cards
- Troubleshooting guide

✅ **Easy Rollback**
- Helm versioning
- Simple rollback command
- Zero-downtime updates

✅ **Secure**
- Secrets management
- ConfigMap separation
- RBAC ready

---

## 📊 Deployment Time Estimates

| Task | Time |
|------|------|
| Get Slack credentials | 5 min |
| Verify cluster access | 2 min |
| Run deployment script | 3-5 min |
| Wait for pods ready | 2-3 min |
| Run health checks | 2 min |
| Test endpoints | 5-10 min |
| **TOTAL** | **~20-30 min** |

---

## 🎉 You're Ready!

Everything needed for production deployment is now ready:

```
✅ Complete Helm chart
✅ Automated deployment scripts
✅ FTC cluster configuration
✅ Comprehensive documentation
✅ Pre-deployment checklist
✅ Troubleshooting guide
✅ API endpoints verified
✅ Database migrations ready
```

**Next Step:** Follow the Quick Start section above or review `helm/HELM_DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

---

**Generated:** 2026-06-26  
**Repository:** https://github.com/sabya610/rag_app  
**Status:** Ready for Production Deployment ✅
