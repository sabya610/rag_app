# Helm Deployment Guide - RAG App Slack Integration
## FTC AIE 1.1.1 Kubernetes Cluster

**Document Version:** 1.0  
**Created:** 2026-06-26  
**Target Cluster:** FTC AIE 1.1.1  
**Deployment Type:** Kubernetes with Helm 3.x  

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Helm Chart Structure](#helm-chart-structure)
4. [Deployment Methods](#deployment-methods)
5. [Configuration](#configuration)
6. [Post-Deployment Validation](#post-deployment-validation)
7. [Troubleshooting](#troubleshooting)
8. [Rollback & Uninstall](#rollback--uninstall)

---

## Quick Start

### Option 1: Using PowerShell (Windows)

```powershell
# From the rag_app directory
cd c:\Users\malliks\rag_app

# Run the Helm deployment script
.\helm\Deploy-Helm.ps1 -Action deploy

# You'll be prompted for:
# - Slack Bot Token
# - Slack Signing Secret
# - Salesforce credentials (optional)
```

### Option 2: Using Bash (Linux/Mac/WSL)

```bash
cd /path/to/rag_app

# Make script executable
chmod +x helm/deploy-helm.sh

# Run deployment
./helm/deploy-helm.sh deploy
```

### Option 3: Direct Helm Command

```bash
helm upgrade --install rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set slack.botToken=xoxb-YOUR_TOKEN \
  --set slack.signingSecret=YOUR_SECRET \
  --namespace default \
  --wait
```

---

## Prerequisites

### 1. Cluster Access

You must have kubeconfig configured to access FTC AIE 1.1.1 cluster:

```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Expected output: Master and Worker nodes from the cluster
```

### 2. Required Tools

- **Helm 3.x**: https://helm.sh/docs/intro/install/
- **kubectl**: https://kubernetes.io/docs/tasks/tools/
- **Docker** (for building images): https://docs.docker.com/get-docker/

### 3. Slack Credentials

Get from: https://api.slack.com/apps

- **Bot Token**: Format `xoxb-...` (minimum 100+ chars)
- **Signing Secret**: 32+ character string

### 4. Salesforce Credentials (Optional)

- SFDC Username
- SFDC Password
- Security Token (sent to your email)

### 5. Docker Registry Access

If pushing to Docker Hub:

```bash
# Login to Docker registry
docker login docker.io
# Or: docker login your-registry.com
```

---

## Helm Chart Structure

```
helm/
├── rag-app-slack/
│   ├── Chart.yaml                 # Chart metadata
│   ├── values.yaml                # Default values
│   ├── templates/
│   │   ├── deployment.yaml        # Main K8s deployment
│   │   ├── service.yaml           # Service definition
│   │   ├── configmap.yaml         # Configuration
│   │   ├── secret.yaml            # Secrets (Slack, SFDC)
│   │   ├── serviceaccount.yaml    # Service account
│   │   ├── hpa.yaml               # Autoscaling policy
│   │   ├── _helpers.tpl           # Template helpers
│   │   └── NOTES.txt              # Post-install notes
│   └── .helmignore
├── values-ftc-aie.yaml            # FTC cluster-specific values
├── deploy-helm.sh                 # Bash deployment script
├── Deploy-Helm.ps1                # PowerShell deployment script
└── README.md                       # This file
```

### Key Components Deployed

| Component | Type | Purpose |
|-----------|------|---------|
| **rag-app-slack** | Deployment | Main application pods |
| **rag-app-slack** | Service | Kubernetes service (ClusterIP) |
| **rag-app-slack** | ConfigMap | Environment configuration |
| **rag-app-slack-secret** | Secret | Slack & SFDC credentials |
| **PostgreSQL** | StatefulSet (via dependency) | Database backend |
| **HPA** | HorizontalPodAutoscaler | Auto-scaling rules |

---

## Deployment Methods

### Method 1: Automated PowerShell (Recommended for Windows)

```powershell
# From rag_app directory
.\helm\Deploy-Helm.ps1 -Action deploy

# The script will:
# 1. Verify prerequisites (helm, kubectl, cluster access)
# 2. Add Bitnami Helm repo (for PostgreSQL)
# 3. Lint the Helm chart
# 4. Create namespace (if needed)
# 5. Prompt for credentials
# 6. Deploy the release
# 7. Verify deployment
# 8. Show deployment info and endpoints
```

### Method 2: Automated Bash (Recommended for Linux/Mac)

```bash
./helm/deploy-helm.sh deploy

# Same steps as PowerShell, but using bash
```

### Method 3: Manual Helm Commands

```bash
# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Create namespace
kubectl create namespace default

# Lint chart
helm lint ./helm/rag-app-slack

# Deploy with custom values
helm upgrade --install rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set slack.botToken=xoxb-YOUR_BOT_TOKEN \
  --set slack.signingSecret=YOUR_SIGNING_SECRET \
  --set salesforce.username=your_username \
  --set salesforce.password=your_password \
  --set salesforce.securityToken=your_token \
  --namespace default \
  --wait \
  --timeout 5m
```

### Method 4: Using Helm Chart Package

```bash
# Create a release package
helm package ./helm/rag-app-slack

# Deploy from package
helm install rag-app-slack ./rag-app-slack-1.0.0.tgz \
  -f ./helm/values-ftc-aie.yaml \
  --set slack.botToken=xoxb-... \
  --set slack.signingSecret=...
```

---

## Configuration

### Default Values Override

The `values-ftc-aie.yaml` file contains FTC-specific configuration:

```yaml
ragApp:
  replicaCount: 2                    # Number of pod replicas
  image:
    repository: docker.io/your-username/rag-app
    tag: slack-integrated
  resources:
    requests:
      memory: "3Gi"
      cpu: "1500m"
    limits:
      memory: "6Gi"
      cpu: "3000m"

postgresql:
  persistence:
    size: 100Gi                      # Database volume size
    storageClassName: "default"      # Storage class for PVC

slack:
  importLimit: 100                   # Max messages per import
  importDays: 30                     # Days back to import

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 4
  targetCPUUtilizationPercentage: 70
```

### Secret Management

Sensitive values are stored in Kubernetes Secrets:

```bash
# View deployed secrets
kubectl get secrets -n default

# Examine secret (base64 encoded)
kubectl get secret rag-app-slack-secret -n default -o yaml

# Decode secret value
kubectl get secret rag-app-slack-secret -n default -o jsonpath='{.data.slack-bot-token}' | base64 -d

# Update secret after deployment
kubectl set env deployment/rag-app-slack \
  SLACK_BOT_TOKEN=xoxb-new-token \
  -n default
```

### ConfigMap Values

Non-sensitive configuration:

```bash
# View ConfigMap
kubectl get configmap rag-app-slack-config -n default -o yaml

# Update ConfigMap
kubectl patch configmap rag-app-slack-config -n default \
  --type merge -p '{"data":{"slack-import-limit":"200"}}'
```

---

## Post-Deployment Validation

### 1. Check Deployment Status

```bash
# View Helm release status
helm status rag-app-slack -n default

# Get deployment status
kubectl get deployment rag-app-slack -n default -o wide

# Get pod status
kubectl get pods -n default -l app.kubernetes.io/instance=rag-app-slack
```

### 2. Check Pod Logs

```bash
# View logs from all pods
kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack -f

# View logs from specific pod
kubectl logs -n default POD_NAME -f

# View logs from container
kubectl logs -n default POD_NAME -c rag-app-slack -f
```

### 3. Check Service

```bash
# Get service details
kubectl get svc rag-app-slack -n default

# Get service endpoints
kubectl get endpoints rag-app-slack -n default

# Port-forward for testing
kubectl port-forward -n default svc/rag-app-slack 5001:5001
```

### 4. Test Health Endpoints

```bash
# From within cluster
kubectl exec -it POD_NAME -n default -- curl http://localhost:5001/api/slack/stats

# From local machine (after port-forward)
curl http://localhost:5001/api/slack/stats
```

### 5. Test Slack Channels Endpoint

```bash
curl -X GET http://localhost:5001/api/slack/channels \
  -H "Content-Type: application/json"
```

### 6. Check Database Connectivity

```bash
# Get PostgreSQL service
kubectl get svc -n default | grep postgres

# Test database from pod
kubectl exec -it POD_NAME -n default -- psql -h postgres-slack-postgresql -U postgres -d ragdb_slack -c "\dt"
```

---

## Troubleshooting

### Issue: Pods in CrashLoopBackOff

**Symptoms:** Pods continuously restart

**Diagnosis:**
```bash
kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack --tail=50
```

**Common Causes:**
- Missing credentials (Slack token, SFDC password)
- Database not initialized
- Invalid model paths

**Solutions:**
1. Verify secrets are set correctly:
   ```bash
   kubectl get secret rag-app-slack-secret -n default -o yaml
   ```

2. Check environment variables:
   ```bash
   kubectl describe pod POD_NAME -n default | grep -A 20 "Environment"
   ```

3. Re-deploy with correct values:
   ```bash
   helm upgrade rag-app-slack ./helm/rag-app-slack \
     -f ./helm/values-ftc-aie.yaml \
     --set slack.botToken=xoxb-correct-token \
     --set slack.signingSecret=correct-secret \
     --namespace default
   ```

### Issue: Database Connection Errors

**Symptoms:** "Cannot connect to database" in logs

**Diagnosis:**
```bash
# Check PostgreSQL pod
kubectl get pods -n default | grep postgres

# Check PostgreSQL logs
kubectl logs -n default postgres-slack-0
```

**Solutions:**
1. Verify PostgreSQL is running:
   ```bash
   kubectl get statefulset -n default
   ```

2. Check database credentials:
   ```bash
   kubectl get secret rag-app-slack-secret -n default -o jsonpath='{.data.db-pass}' | base64 -d
   ```

3. Restart PostgreSQL:
   ```bash
   kubectl delete pod -n default postgres-slack-0
   ```

### Issue: Slack Import Fails

**Symptoms:** Import endpoint returns error

**Diagnosis:**
```bash
# Test Slack endpoint
kubectl port-forward -n default svc/rag-app-slack 5001:5001
curl -X POST http://localhost:5001/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{"channel_ids": ["C123456"], "days_back": 7}'
```

**Solutions:**
1. Verify Slack bot token:
   ```bash
   kubectl exec -it POD_NAME -n default -- python -c "from slack_sdk import WebClient; c = WebClient(token='xoxb-...'); print(c.auth_test())"
   ```

2. Check bot has channel access:
   - In Slack, add bot to the channel
   - Verify token has `channels:history` scope

3. Check rate limits:
   ```bash
   kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack | grep -i "rate"
   ```

### Issue: Helm Release Stuck

**Symptoms:** `helm upgrade` hangs or times out

**Causes:**
- Pod stuck in Creating/Pending state
- No available resources
- Image pull issues

**Solutions:**
1. Check pod status:
   ```bash
   kubectl describe pod POD_NAME -n default
   ```

2. Check node resources:
   ```bash
   kubectl top nodes
   kubectl top pods -n default
   ```

3. Force rollback:
   ```bash
   helm rollback rag-app-slack 0 -n default
   ```

---

## Rollback & Uninstall

### View Helm Release History

```bash
helm history rag-app-slack -n default
```

### Rollback to Previous Version

```bash
# Rollback to previous release
helm rollback rag-app-slack 0 -n default

# Rollback to specific revision
helm rollback rag-app-slack 2 -n default
```

### Uninstall Helm Release

```bash
# Uninstall (keeps PVC)
helm uninstall rag-app-slack -n default

# Uninstall and delete all resources including PVC
helm uninstall rag-app-slack -n default && \
kubectl delete pvc -n default -l app.kubernetes.io/instance=rag-app-slack
```

### Full Cleanup

```bash
# Delete namespace and all resources
kubectl delete namespace default
kubectl create namespace default

# Or just delete PVCs if keeping namespace
kubectl delete pvc -n default -l app.kubernetes.io/instance=rag-app-slack
```

---

## Advanced Configuration

### Enable Ingress

```bash
helm upgrade rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set ragApp.ingress.enabled=true \
  --set ragApp.ingress.hosts[0].host=rag-app.example.com \
  --set ragApp.ingress.hosts[0].paths[0].path=/ \
  --namespace default
```

### Enable Autoscaling

```bash
helm upgrade rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set ragApp.autoscaling.enabled=true \
  --set ragApp.autoscaling.minReplicas=2 \
  --set ragApp.autoscaling.maxReplicas=10 \
  --namespace default
```

### Increase Resource Limits

```bash
helm upgrade rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set ragApp.resources.requests.memory=4Gi \
  --set ragApp.resources.limits.memory=8Gi \
  --namespace default
```

### Custom Storage Class

```bash
helm upgrade rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set postgresql.primary.persistence.storageClassName=fast-ssd \
  --namespace default
```

---

## Monitoring

### Check Resource Usage

```bash
# Cluster-wide metrics
kubectl top nodes

# Pod-level metrics
kubectl top pods -n default

# Detailed metrics
kubectl get hpa -n default
kubectl describe hpa rag-app-slack -n default
```

### View Events

```bash
# Cluster events
kubectl get events -n default --sort-by='.lastTimestamp'

# Pod-specific events
kubectl describe pod POD_NAME -n default | grep -A 20 "Events:"
```

### Check Database Storage

```bash
# PVC usage
kubectl get pvc -n default
kubectl describe pvc -n default

# Inside PostgreSQL
kubectl exec -it postgres-slack-0 -n default -- psql -U postgres -d ragdb_slack -c "SELECT pg_size_pretty(pg_database_size('ragdb_slack'));"
```

---

## Support & Documentation

### Useful Resources

- **Helm Docs:** https://helm.sh/docs/
- **Kubernetes Docs:** https://kubernetes.io/docs/
- **Slack API:** https://api.slack.com/docs
- **PostgreSQL:** https://www.postgresql.org/docs/
- **RAG App Repo:** https://github.com/sabya610/rag_app

### Common Commands Reference

```bash
# Helm operations
helm install rag-app-slack ./helm/rag-app-slack
helm upgrade rag-app-slack ./helm/rag-app-slack
helm uninstall rag-app-slack
helm status rag-app-slack
helm history rag-app-slack
helm rollback rag-app-slack

# kubectl operations
kubectl get pods -n default -w
kubectl logs -f pod-name
kubectl describe pod pod-name
kubectl exec -it pod-name -- bash
kubectl port-forward svc/rag-app-slack 5001:5001
kubectl scale deployment rag-app-slack --replicas=3

# Cluster information
kubectl cluster-info
kubectl get nodes
kubectl get namespaces
kubectl top nodes && kubectl top pods
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-26 | Initial release |

---

**Last Updated:** 2026-06-26  
**Maintained By:** Deployment Team
