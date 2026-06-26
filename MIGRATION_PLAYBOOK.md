# RAG App Migration Playbook: Existing → PCAI

## 📋 Executive Summary

**Current State:** rag-app deployed with:
- 1 replica (single point of failure)
- Hardcoded credentials in values.yaml (🔴 SECURITY RISK)
- Port 80
- Basic SFDC integration

**Target State (PCAI):** rag-app-slack deployed with:
- 2 replicas with autoscaling (HA)
- Credentials in Kubernetes Secrets (🟢 SECURE)
- Port 5001
- SFDC + Slack integration
- Enhanced security and monitoring

**Migration Risk:** ⚠️ **PORT CHANGE** (80 → 5001)

---

## 🔴 CRITICAL: Credential Export (DO THIS FIRST!)

**⚠️ ACTION REQUIRED IMMEDIATELY:**

### Export SFDC Session ID

```bash
# Current SFDC session ID (hardcoded in existing values.yaml)
# From screenshot: 00Dd0000000bUlK!ARAAQNTwCCIMaNuxRE5rwqR4oRHBAq9_...

# Save this value:
export SFDC_SESSION_ID="00Dd0000000bUlK!ARAAQNTwCCIMaNuxRE5rwqR4oRHBAq9_Z5VZiLdQDH0Qt8UGHhW5mKOVn.RPks4hiG_Wi5SmyjvWvX7cPfyym1GWTdqcWwIy"

# Also check if mounted as file
kubectl get pod -n rag-app -o yaml | grep "/etc/sfdc"
```

### Export Database Password

```bash
# Export existing DB password
export DB_PASSWORD="postgres"  # From existing values.yaml

# Verify in running pod
kubectl exec -n rag-app POD_NAME -- env | grep DB_PASS
```

### Export Slack Credentials (if already configured)

```bash
# Get bot token from running pod
kubectl exec -n rag-app POD_NAME -- env | grep SLACK
```

---

## 📋 Pre-Migration Checklist

- [ ] Export SFDC session ID
- [ ] Export DB password
- [ ] Backup current deployment: `kubectl get deployment rag-app -o yaml > rag-app-backup.yaml`
- [ ] Backup current PVC data (if persistent)
- [ ] Verify PCAI registry access: `kubectl run --rm -it --image=registry.pcai.local/rag-app:slack-integrated --restart=Never test -- echo "OK"`
- [ ] Verify port 5001 availability
- [ ] Verify EZUA/Istio is running: `kubectl get virtualservice -n istio-system`
- [ ] Schedule maintenance window (app will be down 5-10 minutes)
- [ ] Notify users of deployment

---

## 🔧 Migration Steps

### Phase 1: Prepare PCAI (10 minutes)

**1. Create PCAI namespace**
```bash
kubectl create namespace rag-app-sfdc-slack
kubectl label namespace rag-app-sfdc-slack name=rag-app-sfdc-slack
```

**2. Create Kubernetes Secrets with exported credentials**
```bash
# PostgreSQL secret with exported password
kubectl create secret generic postgres-secret \
  --from-literal=postgres-password=YOUR_EXPORTED_DB_PASSWORD \
  --from-literal=password=YOUR_EXPORTED_DB_PASSWORD \
  -n rag-app-sfdc-slack

# SFDC credentials (session ID from existing)
kubectl create secret generic sfdc-credentials \
  --from-literal=sessionId=YOUR_EXPORTED_SESSION_ID \
  -n rag-app-sfdc-slack

# Slack credentials (if available)
kubectl create secret generic slack-credentials \
  --from-literal=botToken=xoxb-YOUR_TOKEN \
  --from-literal=signingSecret=YOUR_SECRET \
  -n rag-app-sfdc-slack
```

**3. Verify secrets created**
```bash
kubectl get secrets -n rag-app-sfdc-slack
# Output should show: postgres-secret, sfdc-credentials, slack-credentials
```

### Phase 2: Deploy New Version (5 minutes)

**1. Deploy with merged values**
```bash
# From releases directory with tar.gz package:
tar -xzf rag-app-slack-deployment-*.tar.gz
cd rag-app-slack-deployment-*/helm

# Deploy
helm install rag-app-slack ./rag-app-slack \
  -f values-pcai-merged.yaml \
  -n rag-app-sfdc-slack
```

Or using local files:
```bash
helm install rag-app-slack ./helm/rag-app-slack \
  -f helm/values-pcai-merged.yaml \
  -n rag-app-sfdc-slack
```

**2. Wait for readiness**
```bash
# Watch pod startup
kubectl get pods -n rag-app-sfdc-slack -w

# Should see:
# rag-app-slack-0    1/1     Running
# rag-app-slack-1    1/1     Running (after first is healthy)
# postgres-slack-0   1/1     Running
```

**3. Check health**
```bash
# Wait for ready status
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/instance=rag-app-slack \
  -n rag-app-sfdc-slack \
  --timeout=300s

# View logs for errors
kubectl logs -n rag-app-sfdc-slack \
  -l app.kubernetes.io/instance=rag-app-slack -f
```

### Phase 3: Verify Functionality (5 minutes)

**1. Port-forward for testing**
```bash
kubectl port-forward -n rag-app-sfdc-slack \
  svc/rag-app-slack 5001:5001 &

# Test health
curl http://localhost:5001/health

# Test SFDC integration
curl http://localhost:5001/api/sfdc/status

# Test Slack (if configured)
curl http://localhost:5001/api/slack/channels

# Stop port-forward
kill %1
```

**2. Check VirtualService (EZUA integration)**
```bash
kubectl get virtualservice -n rag-app-sfdc-slack
# Should show: rag-app-slack

# Verify endpoint
kubectl describe virtualservice rag-app-slack -n rag-app-sfdc-slack
```

**3. Verify database migration**
```bash
# Check if DB populated
kubectl exec -n rag-app-sfdc-slack postgres-slack-0 \
  -- psql -U postgres ragdb_slack -c "\dt"

# Should show existing tables from old deployment
```

### Phase 4: Traffic Switchover (5 minutes)

**Option A: Update ingress/DNS**
```bash
# Get new ingress IP
kubectl get ingress -n rag-app-sfdc-slack
# Update DNS to point to new IP

# Test new endpoint
curl https://rag-app.pcai.local/health
```

**Option B: Update service routing**
```bash
# If using service selector, update old deployment selector
# kubectl patch svc rag-app -p '{"spec":{"selector":{"app":"rag-app-slack"}}}'
```

**Option C: Canary (gradual traffic shift)**
```bash
# Use Istio traffic splitting to gradually shift 10% → 50% → 100%
# See CANARY_DEPLOYMENT.md for details
```

### Phase 5: Decommission Old Version (Optional)

**Only after verification period (1-2 hours):**

```bash
# Remove old deployment
kubectl delete deployment rag-app -n rag-app

# Remove old service (if not needed)
kubectl delete svc rag-app -n rag-app

# Cleanup old namespace
kubectl delete namespace rag-app
```

---

## ⚠️ ROLLBACK PLAN

**If something breaks:**

```bash
# 1. Scale down new deployment
kubectl scale deployment rag-app-slack --replicas=0 \
  -n rag-app-sfdc-slack

# 2. Restore old deployment
kubectl apply -f rag-app-backup.yaml -n rag-app

# 3. Update DNS/ingress back to old service
# OR restore service selector

# 4. Investigate issues in dev environment
kubectl logs rag-app-slack-0 -n rag-app-sfdc-slack

# 5. After fix, retry migration
```

---

## 🔍 Comparison: Old vs New

| Aspect | Old (Current) | New (PCAI) | Change |
|--------|---------------|-----------|--------|
| Replicas | 1 | 2 | ✅ Better HA |
| Port | 80 | 5001 | ⚠️ Must update |
| Security | Hardcoded secrets | K8s Secrets | ✅ Better |
| Autoscaling | None | 2-5 | ✅ Better scaling |
| Models | Not persistent | Persistent | ✅ Better reliability |
| Monitoring | Basic | Enhanced | ✅ Better observability |
| SFDC | Yes | Yes | ✅ Same functionality |
| Slack | No | Yes | ✅ New feature |
| EZUA | Yes | Yes | ✅ Same integration |

---

## 📊 Expected Metrics After Migration

```bash
# Check pod resource usage
kubectl top pods -n rag-app-sfdc-slack
NAME                     CPU      MEMORY
rag-app-slack-0         412m     3821Mi
rag-app-slack-1         398m     3756Mi
postgres-slack-0        89m      1242Mi

# Expected autoscaling behavior
# - CPU avg > 70% → scale to 3 pods
# - Memory avg > 75% → scale to 3 pods
# - CPU avg < 50% for 5 min → scale to 2 pods
```

---

## 🎯 Success Criteria

✅ Migration considered successful when:

- [ ] Both app pods running and healthy
- [ ] SFDC integration functional (can query cases)
- [ ] Slack integration functional (if configured)
- [ ] Database accessible and populated
- [ ] Ingress/DNS resolving correctly
- [ ] Health check endpoint responding (200 OK)
- [ ] No errors in pod logs (only INFO level)
- [ ] Autoscaling triggers correctly
- [ ] VirtualService/EZUA integration working
- [ ] Resource usage within expected limits

---

## 📝 Post-Migration Tasks

1. **Monitor logs for 30 minutes**
   ```bash
   kubectl logs -n rag-app-sfdc-slack \
     -l app.kubernetes.io/instance=rag-app-slack -f
   ```

2. **Update documentation**
   - Update runbooks to reference new port (5001)
   - Update dashboards to use new namespace
   - Update alerting rules

3. **Review performance**
   ```bash
   kubectl top pods -n rag-app-sfdc-slack
   kubectl get hpa -n rag-app-sfdc-slack
   ```

4. **Schedule routine tasks**
   - Backup database
   - Test disaster recovery
   - Configure backups for new PVCs

---

## 🆘 Troubleshooting

### Pods stuck in CrashLoopBackOff
```bash
kubectl logs -n rag-app-sfdc-slack POD_NAME
# Look for:
# - Secret not found: Verify secrets created
# - DB connection failed: Check postgres pod
# - Model files not found: Check model persistence PVC
```

### SFDC connection fails
```bash
# Check session ID is valid
kubectl exec -n rag-app-sfdc-slack POD_NAME \
  -- cat /etc/sfdc/sid.txt

# Check if session expired
# Note: SFDC sessions expire after 8 hours, may need refresh
```

### Port 5001 not accessible
```bash
# Check ingress configuration
kubectl get ingress -n rag-app-sfdc-slack -o yaml

# Check service port mapping
kubectl get svc -n rag-app-sfdc-slack

# Verify network policy allows traffic
kubectl get networkpolicy -n rag-app-sfdc-slack
```

### Database size grows unexpectedly
```bash
# Check PVC usage
kubectl exec -n rag-app-sfdc-slack postgres-slack-0 \
  -- du -sh /var/lib/postgresql/data

# If growing too fast, may need to truncate old data
# Consult DBA before clearing data
```

---

## 📞 Escalation Contacts

- **PCAI Cluster Admin:** [Contact info]
- **Kubernetes Support:** [Contact info]
- **Database DBA:** [Contact info]
- **SFDC Admin:** [Contact info]

---

## ✅ Migration Checklist Summary

**Before Migration:**
- [ ] Export credentials
- [ ] Backup current deployment
- [ ] Verify PCAI resources
- [ ] Schedule maintenance window

**During Migration:**
- [ ] Create PCAI namespace
- [ ] Create Kubernetes secrets
- [ ] Deploy new Helm chart
- [ ] Verify health endpoints
- [ ] Run test suite
- [ ] Switch traffic

**After Migration:**
- [ ] Monitor logs
- [ ] Verify functionality
- [ ] Update documentation
- [ ] Schedule backups
- [ ] Optional: Decommission old version

---

**Estimated Total Time:** 30 minutes (+ monitoring)  
**Rollback Time:** 5 minutes  
**Risk Level:** Medium (port change, database change)  
**Recommended Window:** Off-peak hours (low traffic)
