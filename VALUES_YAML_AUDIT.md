# ⚠️ VALUES.YAML ISSUES & FIXES FOR PCAI DEPLOYMENT

## CRITICAL ISSUES

### 1. **DUPLICATE `models:` SECTION** ❌ YAML INVALID
**Problem:** Two `models:` keys defined (lines 73-76 and 79-83)
```yaml
# CURRENT (BROKEN):
models:
  embeddingModel: "models/embedding/all-MiniLM-L6-v2"
  llmModel: "models/llama-2-7b-chat.Q4_K_M.gguf"
  embeddingDimension: 384

models:  # ← DUPLICATE KEY!
  persistence:
    enabled: false
```

**Fix:** Merge into single section
```yaml
models:
  embeddingModel: "models/embedding/all-MiniLM-L6-v2"
  llmModel: "models/llama-2-7b-chat.Q4_K_M.gguf"
  embeddingDimension: 384
  persistence:
    enabled: false
    size: 20Gi
    storageClassName: "standard"
    mountPath: /app/models
```

---

## SECURITY ISSUES 🔒

### 2. **Hardcoded Database Password**
**Risk:** Database credentials visible in plain text
```yaml
postgresql:
  auth:
    password: postgres123  # ← EXPOSED!
```

**Fix:** Use Kubernetes Secrets
```yaml
postgresql:
  auth:
    existingSecret: "postgres-secret"
    secretKeys:
      adminPasswordKey: "postgres-password"
      userPasswordKey: "password"
    database: ragdb_slack
```

Then create secret:
```bash
kubectl create secret generic postgres-secret \
  --from-literal=postgres-password=YOUR_SECURE_PASSWORD \
  --from-literal=password=YOUR_SECURE_PASSWORD
```

### 3. **Hardcoded Salesforce Token**
```yaml
securityToken: "MDBEZDAwMDAwMDBiVWxLIUFSQUFRSlFwQWRXUE1nMlUxaW9HcTJsNVRJTjZ5aGxOOFhjR2N0VkhKZGh4bzY4QTRMWUFWLjd6VmFHSktrZ1BLNnhxOHVocEtRUk0yTUIwcnJpQ2JqZWpycmRpTE9aNgo="  # ← EXPOSED!
```

**Fix:** Move to secret
```yaml
salesforce:
  credentialsSecret: "sfdc-credentials"  # Reference secret instead
  enabled: true
```

### 4. **Root User Execution**
```yaml
podSecurityContext:
  runAsNonRoot: false
  runAsUser: 0  # ← RUNNING AS ROOT!

securityContext:
  allowPrivilegeEscalation: true  # ← DANGEROUS!
```

**Fix:** Run as non-root
```yaml
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
    add:
      - NET_BIND_SERVICE  # Only if needed
```

---

## RESOURCE & PERFORMANCE ISSUES 📊

### 5. **Insufficient Memory for ML Models**
**Problem:** 2Gi request won't handle embedding + LLM model
- Sentence Transformer (all-MiniLM): ~300MB
- Llama 2 7B Q4: ~4-5GB
- App overhead: 500MB+

**Fix:** Increase resources
```yaml
resources:
  requests:
    memory: "6Gi"      # Was 2Gi
    cpu: "2000m"       # Was 1000m
  limits:
    memory: "8Gi"      # Was 4Gi
    cpu: "3000m"       # Was 2000m
```

### 6. **Autoscaling Disabled**
```yaml
autoscaling:
  enabled: false  # ← SHOULD BE ENABLED FOR PRODUCTION
```

**Fix:** Enable autoscaling
```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 75
```

---

## STORAGE & DATABASE ISSUES 💾

### 7. **Storage Class May Not Exist**
```yaml
storageClassName: "default"  # Might not exist in PCAI
```

**Fix:** Check available classes and use correct one
```bash
kubectl get storageclass
```

Typical PCAI options:
```yaml
postgresql:
  primary:
    persistence:
      storageClassName: "fast-ssd"  # Or "standard", "nfs-provisioner"
      size: 100Gi                    # Increased from 50Gi for production

models:
  persistence:
    storageClassName: "standard"
    size: 30Gi                       # Increased for models
```

### 8. **Empty Credentials Not Handled**
```yaml
slack:
  botToken: ""           # ← REQUIRED, NOT SET!
  signingSecret: ""

salesforce:
  username: ""           # ← REQUIRED, NOT SET!
```

**Fix:** Add validation/defaults
```yaml
slack:
  enabled: true
  # Note: Must be set via --set or values-override
  # botToken: xoxb-YOUR_TOKEN_HERE (use secrets!)
  # signingSecret: YOUR_SECRET_HERE (use secrets!)
  importLimit: 100
  importDays: 30

salesforce:
  enabled: false  # Set to true only if configured
  # credentials via secret reference
```

---

## INGRESS & NETWORKING ISSUES 🌐

### 9. **Ingress Disabled but Needed**
```yaml
ingress:
  enabled: false  # ← PRODUCTION NEEDS THIS
```

**Fix:** Enable and configure for PCAI
```yaml
ingress:
  enabled: true
  className: "pcai-ingress"  # Check: kubectl get ingressclass
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: rag-app.pcai.local  # Or your domain
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: rag-app-tls
      hosts:
        - rag-app.pcai.local
```

---

## HEALTH CHECKS & INITIALIZATION ⚕️

### 10. **Health Check Endpoint Might Fail**
```yaml
livenessProbe:
  httpGet:
    path: /api/slack/stats  # ← Might not exist if app not initialized
```

**Fix:** Use better health check
```yaml
livenessProbe:
  httpGet:
    path: /health  # More reliable
    port: 5001
  initialDelaySeconds: 60  # Increased from 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 5001
  initialDelaySeconds: 15  # Increased from 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

---

## NAMESPACE & RBAC ISSUES 👤

### 11. **Service Account Not Configured**
```yaml
serviceAccount:
  create: true
  name: rag-app-slack
  # Missing: annotations, automountServiceAccountToken
```

**Fix:** Add proper RBAC
```yaml
serviceAccount:
  create: true
  name: rag-app-slack
  automountServiceAccountToken: true
  annotations:
    description: "RAG App Slack Integration Service Account"

# Add this section to templates/rbac.yaml:
# ClusterRole for reading config/secrets
# RoleBinding to namespace
```

---

## PCAI-SPECIFIC RECOMMENDATIONS 🔧

### For PCAI Deployment:

1. **Storage:** Use PCAI's storage provisioner
```yaml
# Find available storage:
kubectl get storageclass -o yaml | grep 'name:'

# Update values:
postgresql:
  primary:
    persistence:
      storageClassName: "pcai-storage"  # Or actual PCAI class
  metrics:
    enabled: true  # Enable for monitoring

models:
  persistence:
    storageClassName: "pcai-storage"
    enabled: true  # Enable for persistent models
```

2. **Registry:** Use PCAI registry
```yaml
ragApp:
  image:
    repository: registry.pcai.local/rag-app  # Update registry
    tag: slack-integrated
    pullPolicy: IfNotPresent
```

3. **Network Policies:** Add for security
```yaml
networkPolicy:
  enabled: true
  policyTypes:
    - Ingress
    - Egress
```

4. **Resource Quotas:** Set for namespace
```yaml
resourceQuota:
  enabled: true
  limits:
    requests:
      memory: "32Gi"
      cpu: "8000m"
```

---

## CORRECTED VALUES.YAML SNIPPET

```yaml
global:
  environment: production
  clusterName: "PCAI"

ragApp:
  enabled: true
  replicaCount: 2
  
  image:
    repository: registry.pcai.local/rag-app
    tag: slack-integrated
    pullPolicy: IfNotPresent
  
  resources:
    requests:
      memory: "6Gi"        # FIXED: Increased
      cpu: "2000m"         # FIXED: Increased
    limits:
      memory: "8Gi"        # FIXED: Increased
      cpu: "3000m"         # FIXED: Increased
  
  autoscaling:
    enabled: true         # FIXED: Enabled
    minReplicas: 2
    maxReplicas: 5
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 75
  
  healthCheck:
    enabled: true
    livenessProbe:
      httpGet:
        path: /health     # FIXED: Better endpoint
        port: 5001
      initialDelaySeconds: 60  # FIXED: Increased
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health
        port: 5001
      initialDelaySeconds: 15
      periodSeconds: 5

postgresql:
  enabled: true
  auth:
    existingSecret: "postgres-secret"  # FIXED: Use secret
    database: ragdb_slack
  
  primary:
    persistence:
      enabled: true
      size: 100Gi         # FIXED: Increased
      storageClassName: "pcai-storage"  # FIXED: Use PCAI storage

models:
  embeddingModel: "models/embedding/all-MiniLM-L6-v2"
  llmModel: "models/llama-2-7b-chat.Q4_K_M.gguf"
  embeddingDimension: 384
  persistence:
    enabled: true        # FIXED: Enable persistence
    size: 30Gi
    storageClassName: "pcai-storage"
    mountPath: /app/models

slack:
  enabled: true
  botToken: ""          # FIXED: Use secret reference
  signingSecret: ""     # FIXED: Use secret reference
  importLimit: 100
  importDays: 30

salesforce:
  enabled: false        # FIXED: Default to false, enable when configured
  credentialsSecret: "sfdc-credentials"  # FIXED: Use secret

ingress:
  enabled: true         # FIXED: Enable for production
  className: "pcai-ingress"
  hosts:
    - host: rag-app.pcai.local
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: rag-app-tls
      hosts:
        - rag-app.pcai.local

podSecurityContext:
  runAsNonRoot: true    # FIXED: Security
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false  # FIXED: Security
  capabilities:
    drop:
      - ALL

namespace: rag-app-sfdc-slack

serviceAccount:
  create: true
  name: rag-app-slack
  automountServiceAccountToken: true
```

---

## DEPLOYMENT STEPS FOR PCAI

1. **Create secrets first:**
```bash
kubectl create secret generic postgres-secret \
  --from-literal=postgres-password=$(openssl rand -base64 32) \
  --from-literal=password=$(openssl rand -base64 32)

kubectl create secret generic sfdc-credentials \
  --from-literal=username=YOUR_SFDC_USER \
  --from-literal=password=YOUR_SFDC_PASSWORD \
  --from-literal=securityToken=YOUR_TOKEN \
  --from-literal=clientId=YOUR_CLIENT_ID \
  --from-literal=clientSecret=YOUR_CLIENT_SECRET
```

2. **Install with corrected values:**
```bash
helm install rag-app-slack ./helm/rag-app-slack \
  -f values-pcai.yaml \
  --set slack.botToken=xoxb-YOUR_TOKEN \
  --set slack.signingSecret=YOUR_SECRET
```

3. **Verify:**
```bash
kubectl get pods -n rag-app-sfdc-slack
kubectl get pvc -n rag-app-sfdc-slack
kubectl get ingress -n rag-app-sfdc-slack
```
