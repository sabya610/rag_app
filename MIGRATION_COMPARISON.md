# Comparison: Existing rag-app vs PCAI Deployment

## 📊 Configuration Comparison Matrix

| Feature | Existing Deployed | PCAI (Proposed) | Status | Migration Note |
|---------|-------------------|-----------------|--------|-----------------|
| **Registry** | sabya610/rag-app | registry.pcai.local/rag-app | ⚠️ UPDATE | Need PCAI registry path |
| **Replica Count** | 1 | 2 | ✅ IMPROVEMENT | Better HA |
| **Port** | 80 | 5001 | ⚠️ CHANGE | Port mismatch |
| **Service Type** | ClusterIP | ClusterIP | ✅ SAME | No change needed |
| **Autoscaling** | None | Enabled (2-5) | ✅ IMPROVEMENT | Better scaling |
| **Memory Request** | 4Gi | 6Gi | ✅ IMPROVEMENT | Better for ML models |
| **Memory Limit** | 8Gi | 8Gi | ✅ SAME | No change needed |
| **CPU Request** | 2 | 2000m | ✅ SAME | Equivalent |
| **CPU Limit** | 4 | 3000m | ✅ IMPROVEMENT | More efficient |
| **Ingress** | Disabled | Enabled with TLS | ✅ IMPROVEMENT | Need domain config |
| **Health Check** | Not visible | /health endpoint | ✅ NEW | Better monitoring |
| **PostgreSQL** | Enabled (5Gi) | Enabled (100Gi) | ⚠️ EXPANSION | Larger DB for production |
| **Model Persistence** | Not enabled | Enabled (30Gi) | ✅ NEW | Better reliability |
| **Pod Security** | Root (0) | Non-root (1000) | ✅ IMPROVEMENT | Security hardening |
| **Credentials** | Plain text env vars | Kubernetes Secrets | ✅ CRITICAL FIX | Security critical |
| **SFDC Integration** | Session ID (hardcoded!) | Via secret reference | ✅ CRITICAL FIX | Security critical |
| **EZUA Integration** | VirtualService + AuthPolicy | Not included | ⚠️ MISSING | Need to add! |

---

## 🔴 CRITICAL SECURITY ISSUES IN EXISTING CONFIG

### 1. **Session ID Exposed in Plain Text**
```yaml
# CURRENT (DANGEROUS):
sfdc:
  sessionId: "00Dd0000000bUlK!ARAAQNTwCCIMaNuxRE5rwqR4oRHBAq9_Z5VZiLdQDH0Qt8UGHhW5mKOVn..."
  sessionIdFile: "/etc/sfdc/sid.txt"
```

**Risks:**
- Anyone with cluster access can read the values.yaml
- Session ID appears in pod environment variables
- Credentials are visible in git history if committed

**Fix (PCAI version):**
```yaml
sfdc:
  credentialsSecret: "sfdc-credentials"  # Reference only, not value
```

### 2. **Database Password in Plaintext**
```yaml
# CURRENT (DANGEROUS):
env:
  DB_PASS: "postgres"
postgresql:
  password: postgres
```

**Fix (PCAI version):**
```yaml
postgresql:
  auth:
    existingSecret: "postgres-secret"
```

### 3. **Init Container Not Referenced**
```yaml
initImage:
  repository: sabya610/busybox
  tag: "1.36"
```
**Issue:** Not used anywhere - should be cleaned up or integrated

---

## 🔧 CRITICAL CHANGES NEEDED FOR MIGRATION

### 1. **Port Migration (80 → 5001)**
**Existing:** Service port 80  
**PCAI:** Service port 5001

**Action Required:**
```yaml
# Update deployment to match ingress target
service:
  type: ClusterIP
  port: 5001
  targetPort: 5001
```

**Check:** Existing app listening on which port?
```bash
kubectl get svc rag-app -o yaml | grep port
```

### 2. **Registry Update**
**Existing:** `sabya610/rag-app:latest`  
**PCAI:** `registry.pcai.local/rag-app:slack-integrated`

**Action Required:**
```bash
# Verify PCAI registry is accessible
kubectl run --rm -it --image=registry.pcai.local/rag-app:slack-integrated \
  --restart=Never test -- echo "Registry OK"

# Update values-pcai.yaml with correct registry
ragApp:
  image:
    repository: registry.pcai.local/rag-app  # ← Update this
    tag: slack-integrated
```

### 3. **Add Missing EZUA Integration**
**PCAI version missing:** VirtualService and AuthorizationPolicy

**Action:** Add EZUA section back:
```yaml
# Add to values-pcai.yaml
ezua:
  enabled: true
  virtualService:
    endpoint: "rag-app.${DOMAIN_NAME}"
    istioGateway: "istio-system/ezaf-gateway"
  authorizationPolicy:
    namespace: "istio-system"
    providerName: "oauth2-proxy"
    matchLabels:
      istio: "ingressgateway"
```

---

## 📋 DETAILED MIGRATION GUIDE

### Step 1: Export Existing Credentials

```bash
# Get current SFDC session ID
kubectl get secret sfdc-credentials -o yaml
# OR
kubectl get pod -o yaml | grep SFDC_SESSION

# Get current DB password
kubectl get secret postgres-secret -o yaml
# OR
kubectl get configmap rag-app-config -o yaml | grep DB_PASS
```

### Step 2: Create PCAI Secrets with Existing Values

```bash
# Export existing session ID
EXISTING_SESSION_ID="00Dd0000000bUlK!ARAAQNTwCCIMaNuxRE5rwqR4oRHBAq9_Z5VZiLdQDH0Qt8UGHhW5mKOVn..."
EXISTING_DB_PASS="postgres"

# Create secrets in PCAI namespace
kubectl create secret generic sfdc-credentials \
  --from-literal=sessionId=$EXISTING_SESSION_ID \
  -n rag-app-sfdc-slack

kubectl create secret generic postgres-secret \
  --from-literal=postgres-password=$EXISTING_DB_PASS \
  --from-literal=password=$EXISTING_DB_PASS \
  -n rag-app-sfdc-slack
```

### Step 3: Merge EZUA Configuration

**Create merged values-pcai-with-ezua.yaml:**
```yaml
# Start with values-pcai.yaml and add:

# EZUA Integration (from existing deployment)
ezua:
  enabled: true
  virtualService:
    endpoint: "rag-app.${DOMAIN_NAME}"
    istioGateway: "istio-system/ezaf-gateway"
  authorizationPolicy:
    namespace: "istio-system"
    providerName: "oauth2-proxy"
    matchLabels:
      istio: "ingressgateway"

# Environment variables (from existing)
env:
  DB_HOST: ""  # Will be filled by deployment
  DB_PORT: "5432"
  DB_NAME: "ragdb_slack"
  DB_USER: "postgres"
  # DB_PASS removed - now from secret
  MODEL_PATH: "/app/rag_app/models/llama-2-7b-chat.Q4_K_M.gguf"
  EMBEDDING_MODEL: "/app/rag_app/models/embedding/all-MiniLM-L6-v2"
  SFDC_ENABLED: "true"
  SF_URL: "https://hp.my.salesforce.com"
  SFDC_PRODUCT_QUEUE: "HPE Ezmeral"
  SFDC_PRODUCT_LINE: "CONT PLT SW (RM)"

# SFDC Auth via secret (from existing)
sfdc:
  sessionIdFile: "/etc/sfdc/sid.txt"  # Mount from secret
  # Other auth modes via secret
```

### Step 4: Verify Compatibility

```bash
# Check if Slack features break existing SFDC deployment
# Slack is added feature, should not affect SFDC

# Current behavior should remain:
- SFDC integration: YES (now via secret)
- Database: YES (persistent)
- Models: YES (now persistent in PCAI)
- Port: CHANGES from 80 to 5001

# New features:
- Slack integration: ADDED
- Autoscaling: ADDED
- Ingress: ADDED
- Better security: ADDED
```

### Step 5: Staged Migration

**Option A: New Deployment**
```bash
# Deploy new instance alongside existing
helm install rag-app-slack-v2 ./helm/rag-app-slack \
  -f values-pcai-with-ezua.yaml \
  -n rag-app-sfdc-slack-v2

# Verify it works
# Then redirect traffic and decommission old version
```

**Option B: In-Place Upgrade**
```bash
# Backup existing deployment
kubectl get deployment rag-app -o yaml > rag-app-backup.yaml

# Prepare new values with existing config
# helm upgrade rag-app ./helm/rag-app-slack \
#   -f values-pcai-with-ezua.yaml

# WARNING: This will cause downtime during upgrade
```

---

## ✅ WHAT STAYS THE SAME

```yaml
# These features are preserved:
service:
  type: ClusterIP  # Same

postgresql:
  enabled: true    # Same

models:
  embeddingModel: "/app/rag_app/models/embedding/all-MiniLM-L6-v2"  # Same
  llmModel: "/app/rag_app/models/llama-2-7b-chat.Q4_K_M.gguf"  # Same

sfdc:
  enabled: true    # Same
  SF_URL: "https://hp.my.salesforce.com"  # Same

# EZUA integration (will add back)
ezua:
  enabled: true    # Add back
```

---

## ⚠️ WHAT CHANGES

```yaml
# Port changes
service:
  port: 80 → 5001

# Replicas increase
replicaCount: 1 → 2

# Registry changes
image:
  repository: sabya610/rag-app → registry.pcai.local/rag-app

# Credentials securified
env:
  DB_PASS: removed (→ secret)

sfdc:
  sessionId: removed (→ secret)

# New features added
autoscaling: enabled
ingress: enabled
networkPolicy: enabled
podSecurityContext: hardened
```

---

## 📋 Pre-Migration Checklist

- [ ] Export existing SFDC session ID
- [ ] Export existing DB password
- [ ] Verify PCAI registry accessibility
- [ ] Check if port 5001 is available in cluster
- [ ] Verify EZUA configuration in existing deployment
- [ ] Check ingress configuration (nginx? istio?)
- [ ] Test new image in PCAI registry
- [ ] Plan downtime window if needed
- [ ] Backup current deployment
- [ ] Create secrets in PCAI namespace
- [ ] Deploy and verify new version

---

## 🔄 VALUES.YAML MIGRATION TEMPLATE

Here's a merged version incorporating both existing and new:

```yaml
# RAG App Deployment
ragApp:
  enabled: true
  replicaCount: 2  # CHANGED: from 1
  
  image:
    repository: registry.pcai.local/rag-app  # CHANGED: from sabya610/rag-app
    tag: slack-integrated  # CHANGED: from latest
    pullPolicy: IfNotPresent
  
  service:
    type: ClusterIP
    port: 5001  # CHANGED: from 80
    targetPort: 5001
  
  # Environment from existing deployment
  env:
    DB_HOST: ""
    DB_PORT: "5432"
    DB_NAME: "ragdb_slack"  # CHANGED: from ragdb
    DB_USER: "postgres"
    # DB_PASS: removed - use secret
    MODEL_PATH: "/app/rag_app/models/llama-2-7b-chat.Q4_K_M.gguf"
    EMBEDDING_MODEL: "/app/rag_app/models/embedding/all-MiniLM-L6-v2"
    SFDC_ENABLED: "true"
    SF_URL: "https://hp.my.salesforce.com"
    SFDC_PRODUCT_QUEUE: "HPE Ezmeral"
    SFDC_PRODUCT_LINE: "CONT PLT SW (RM)"
    # NEW Slack variables
    FEATURES_SLACK: "true"
  
  resources:
    requests:
      memory: "6Gi"  # CHANGED: from 4Gi
      cpu: "2000m"  # SAME: 2 cpu
    limits:
      memory: "8Gi"  # SAME
      cpu: "3000m"  # CHANGED: from 4
  
  # NEW: Autoscaling
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 5
    targetCPUUtilizationPercentage: 70
  
  # NEW: Ingress from PCAI
  ingress:
    enabled: true
    className: "pcai-ingress"
    hosts:
      - host: rag-app.pcai.local
        paths:
          - path: /
            pathType: Prefix

# PostgreSQL - from existing
postgresql:
  enabled: true
  image:
    repository: sabya610/pgvector  # Keep existing image
    tag: latest
  auth:
    existingSecret: "postgres-secret"  # NEW: use secret
    database: ragdb_slack  # CHANGED: from ragdb
  primary:
    persistence:
      size: 100Gi  # CHANGED: from 5Gi
      storageClassName: "pcai-storage"

# Slack Integration - NEW
slack:
  enabled: true
  botToken: ""  # Set via secret
  signingSecret: ""  # Set via secret
  importLimit: 100
  importDays: 30

# SFDC from existing + NEW secret approach
salesforce:
  enabled: true
  credentialsSecret: "sfdc-credentials"  # NEW: reference secret
  sessionIdFile: "/etc/sfdc/sid.txt"  # Keep from existing
  username: ""  # From secret
  password: ""  # From secret
  securityToken: ""  # From secret

# NEW: EZUA Integration from existing
ezua:
  enabled: true
  virtualService:
    endpoint: "rag-app.${DOMAIN_NAME}"
    istioGateway: "istio-system/ezaf-gateway"
  authorizationPolicy:
    namespace: "istio-system"
    providerName: "oauth2-proxy"
    matchLabels:
      istio: "ingressgateway"

# Security hardening - NEW
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL

namespace: rag-app-sfdc-slack
```

---

## 🚨 CRITICAL ACTION ITEMS

1. **EXPORT CREDENTIALS IMMEDIATELY** - Before losing existing deployment
   ```bash
   # Get session ID
   kubectl get deployment rag-app -o yaml | grep -i "SESSION"
   
   # Get DB password
   kubectl get secret -o yaml | grep -i "DB_PASS\|password"
   ```

2. **VERIFY REGISTRY** - Can PCAI access docker.io or need custom registry?
   ```bash
   kubectl run --rm -it --image=registry.pcai.local/rag-app:slack-integrated \
     --restart=Never test -- echo "OK"
   ```

3. **TEST PORT CHANGE** - Is port 5001 accessible in PCAI?
   ```bash
   # Deploy test pod on 5001 and verify connectivity
   ```

4. **MERGE EZUA CONFIG** - Don't lose VirtualService configuration

---

## 📞 Next Steps

1. Confirm registry path for PCAI (`registry.pcai.local/rag-app` or different?)
2. Confirm port (80 vs 5001 compatibility)
3. Confirm EZUA integration requirements
4. Create merged values-pcai-combined.yaml
5. Execute migration plan
