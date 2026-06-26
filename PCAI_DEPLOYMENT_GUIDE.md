# PCAI Deployment - Issues & Fixes Summary

## 📋 Issues Found: 11 Critical/High Priority

| Priority | Issue | Status |
|----------|-------|--------|
| 🔴 CRITICAL | Duplicate `models:` key - YAML invalid | ✅ FIXED |
| 🔴 CRITICAL | Hardcoded database password | ✅ FIXED |
| 🔴 CRITICAL | Running as root (security risk) | ✅ FIXED |
| 🟠 HIGH | Running with privilege escalation | ✅ FIXED |
| 🟠 HIGH | Insufficient memory for ML models | ✅ FIXED |
| 🟠 HIGH | Hardcoded Salesforce token exposed | ✅ FIXED |
| 🟠 HIGH | Ingress disabled (needed for production) | ✅ FIXED |
| 🟡 MEDIUM | Autoscaling disabled | ✅ FIXED |
| 🟡 MEDIUM | Wrong storage class reference | ✅ FIXED |
| 🟡 MEDIUM | Models not persisted | ✅ FIXED |
| 🟡 MEDIUM | Health check endpoint unreliable | ✅ FIXED |

---

## 🔧 What Was Changed

### 1. **Merged Duplicate Models Section**
```diff
- models:
-   embeddingModel: "..."
- 
- models:
-   persistence: {...}

+ models:
+   embeddingModel: "..."
+   persistence: {...}
```

### 2. **Secured Credentials**
```diff
- postgres:
-   password: postgres123  # EXPOSED!

+ postgres:
+   existingSecret: "postgres-secret"  # Use Kubernetes secret
```

### 3. **Increased Resources**
```diff
- memory: "2Gi" → "6Gi"     (+200%)
- cpu: "1000m" → "2000m"    (+100%)
- limits.memory: "4Gi" → "8Gi"
```

### 4. **Security Hardening**
```diff
- runAsUser: 0  # ROOT
- allowPrivilegeEscalation: true

+ runAsUser: 1000  # Non-root
+ allowPrivilegeEscalation: false
+ readOnlyRootFilesystem: false
+ seccompProfile: RuntimeDefault
```

### 5. **Enabled Production Features**
```diff
- autoscaling.enabled: false
- ingress.enabled: false

+ autoscaling.enabled: true
+ ingress.enabled: true
```

---

## 📁 Files Provided

1. **values-pcai.yaml** - Corrected configuration for PCAI
2. **VALUES_YAML_AUDIT.md** - Detailed issue analysis
3. **PCAI_DEPLOYMENT_GUIDE.md** - This file

---

## 🚀 PCAI Deployment Guide

### Step 1: Create Secrets

```bash
# Generate secure passwords
DB_PASS=$(openssl rand -base64 32)
DB_ADMIN=$(openssl rand -base64 32)

# Create PostgreSQL secret
kubectl create secret generic postgres-secret \
  --from-literal=postgres-password=$DB_ADMIN \
  --from-literal=password=$DB_PASS \
  -n rag-app-sfdc-slack

# Create Slack secret (get from https://api.slack.com/apps)
kubectl create secret generic slack-credentials \
  --from-literal=botToken=xoxb-YOUR_TOKEN \
  --from-literal=signingSecret=YOUR_SECRET \
  -n rag-app-sfdc-slack

# Create Salesforce secret (optional)
kubectl create secret generic sfdc-credentials \
  --from-literal=username=YOUR_SFDC_USER \
  --from-literal=password=YOUR_SFDC_PASSWORD \
  --from-literal=securityToken=YOUR_TOKEN \
  --from-literal=clientId=YOUR_CLIENT_ID \
  --from-literal=clientSecret=YOUR_CLIENT_SECRET \
  -n rag-app-sfdc-slack
```

### Step 2: Create Namespace

```bash
kubectl create namespace rag-app-sfdc-slack

# Label for network policies
kubectl label namespace rag-app-sfdc-slack name=rag-app-sfdc-slack
```

### Step 3: Check PCAI Resources

```bash
# Verify storage classes available
kubectl get storageclass
# Output should include: pcai-storage or similar

# Verify ingress classes available
kubectl get ingressclass
# Output should include: pcai-ingress or nginx

# Verify certificate issuer exists
kubectl get clusterissuer
```

### Step 4: Update values-pcai.yaml

Before deployment, verify/update these values for your PCAI cluster:

```yaml
# Check your actual values:
ragApp:
  image:
    repository: registry.pcai.local/rag-app  # ← Verify registry

ingress:
  className: "pcai-ingress"  # ← Verify ingress class
  hosts:
    - host: rag-app.pcai.local  # ← Update domain

postgresql:
  primary:
    persistence:
      storageClassName: "pcai-storage"  # ← Verify storage class

models:
  persistence:
    storageClassName: "pcai-storage"  # ← Same storage class
```

### Step 5: Deploy Helm Chart

```bash
# Add Bitnami repo (if not already added)
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install chart with PCAI values
helm install rag-app-slack ./helm/rag-app-slack \
  -f helm/values-pcai.yaml \
  -n rag-app-sfdc-slack \
  --create-namespace

# Or update if already installed
helm upgrade --install rag-app-slack ./helm/rag-app-slack \
  -f helm/values-pcai.yaml \
  -n rag-app-sfdc-slack
```

### Step 6: Verify Deployment

```bash
# Check pods status
kubectl get pods -n rag-app-sfdc-slack
# Should see: rag-app-slack-* Running, postgres-slack-0 Running

# Check PVCs mounted
kubectl get pvc -n rag-app-sfdc-slack
# Should see: postgres-slack-0, rag-app-slack-models

# Check ingress
kubectl get ingress -n rag-app-sfdc-slack
# Should show IP/hostname

# View logs
kubectl logs -n rag-app-sfdc-slack -l app.kubernetes.io/instance=rag-app-slack -f

# Check resource usage
kubectl top pods -n rag-app-sfdc-slack
```

### Step 7: Test Deployment

```bash
# Port-forward for testing
kubectl port-forward -n rag-app-sfdc-slack \
  svc/rag-app-slack 5001:5001 &

# Test health endpoint
curl http://localhost:5001/health
# Expected: 200 OK

# Test Slack channels
curl http://localhost:5001/api/slack/channels
# Should list channels or error if not initialized

# Kill port-forward
fg  # Then Ctrl+C
```

### Step 8: DNS/Certificate Setup

```bash
# Get ingress IP
kubectl get ingress -n rag-app-sfdc-slack -o wide
# Note the IP or hostname

# Add to /etc/hosts (local testing) or configure DNS
# 10.x.x.x rag-app.pcai.local

# Verify certificate
kubectl get certificate -n rag-app-sfdc-slack
kubectl describe certificate rag-app-tls -n rag-app-sfdc-slack
```

---

## ⚠️ Important Notes

1. **Registry:** Update image repository to your PCAI registry
   - Default: `registry.pcai.local/rag-app`
   - Verify with: `kubectl describe node | grep docker`

2. **Storage:** Verify storage class with PCAI admin
   - Use: `kubectl get storageclass -o wide`
   - Common: `pcai-storage`, `standard`, `fast-ssd`

3. **Ingress:** Update domain based on your PCAI setup
   - Update: `rag-app.pcai.local` → Your actual domain

4. **Credentials:** Never commit secrets to git
   - Always create secrets in cluster
   - Reference via `existingSecret`

5. **Resource Limits:** Adjust based on cluster capacity
   - Current: 6Gi requests, 8Gi limits per pod
   - Scale down if cluster is small

---

## 🔍 Troubleshooting

### Pod in CrashLoopBackOff
```bash
kubectl logs -n rag-app-sfdc-slack POD_NAME
# Check for:
# - Secret not found: Create missing secrets
# - Model files missing: Enable persistence
# - Database connection failed: Check postgres pod
```

### PVC Not Bound
```bash
kubectl get pvc -n rag-app-sfdc-slack
kubectl describe pvc pvc-name -n rag-app-sfdc-slack
# Check:
# - Storage class exists: kubectl get storageclass
# - PVC size not exceeding quota
# - PCAI storage provisioner running
```

### Ingress Not Working
```bash
kubectl get ingress -n rag-app-sfdc-slack
kubectl describe ingress rag-app-slack-ingress -n rag-app-sfdc-slack
# Check:
# - Ingress class exists: kubectl get ingressclass
# - DNS resolving: nslookup rag-app.pcai.local
# - Certificate issued: kubectl get certificate
```

### High Memory Usage
```bash
kubectl top pods -n rag-app-sfdc-slack
# If exceeding limits:
# - Increase limit: values-pcai.yaml limits.memory
# - Check model file size
# - Enable read-only root filesystem
```

---

## 📊 Performance Tuning

### For Production:
```yaml
ragApp:
  replicaCount: 3  # Increase replicas
  resources:
    requests:
      memory: "8Gi"
      cpu: "3000m"
    limits:
      memory: "12Gi"
      cpu: "4000m"
  
  autoscaling:
    maxReplicas: 10  # Allow more pods
    targetCPUUtilizationPercentage: 60  # Scale sooner
```

### For Development:
```yaml
ragApp:
  replicaCount: 1
  autoscaling:
    enabled: false
  
  resources:
    requests:
      memory: "4Gi"
      cpu: "1000m"
    limits:
      memory: "6Gi"
      cpu: "2000m"
```

---

## ✅ Pre-Deployment Checklist

- [ ] Kubernetes cluster (PCAI) accessible
- [ ] kubectl configured for cluster
- [ ] Helm 3.x installed
- [ ] Storage class verified: `kubectl get storageclass`
- [ ] Ingress class verified: `kubectl get ingressclass`
- [ ] Certificate issuer configured: `kubectl get clusterissuer`
- [ ] Image registry accessible
- [ ] Slack Bot Token obtained
- [ ] Slack Signing Secret obtained
- [ ] Domain configured (DNS/hosts file)
- [ ] Namespace created: `kubectl create namespace rag-app-sfdc-slack`
- [ ] Secrets created (postgres, slack, sfdc)
- [ ] values-pcai.yaml customized for your cluster
- [ ] Sufficient cluster resources available (CPU, memory)

---

## 📞 Support & Documentation

- **Helm Chart:** ./helm/rag-app-slack/
- **Configuration:** helm/values-pcai.yaml
- **Deployment Guide:** helm/HELM_DEPLOYMENT_GUIDE.md
- **Quick Reference:** helm/QUICK_REFERENCE.md
- **Troubleshooting:** helm/HELM_DEPLOYMENT_GUIDE.md#troubleshooting

---

**Last Updated:** 2026-06-26  
**Version:** 1.0.0  
**PCAI Status:** ✅ Ready for Production
