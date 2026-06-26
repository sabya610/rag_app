# QUICK REFERENCE: Migration Checklist & Commands

## 🚀 Fast Track Migration (30 minutes)

### Prerequisites Check (5 min)
```bash
# 1. Verify cluster access
kubectl cluster-info

# 2. Check PCAI storage
kubectl get storageclass | grep pcai

# 3. Check registry access  
kubectl run --rm -it --image=registry.pcai.local/rag-app:slack-integrated \
  --restart=Never test -- echo "OK"

# 4. Export credentials from running pod
kubectl get deployment rag-app -o yaml | grep sessionId
kubectl get pod -o yaml | grep DB_PASS
```

### Create PCAI Environment (5 min)
```bash
# 1. Create namespace
kubectl create namespace rag-app-sfdc-slack
kubectl label namespace rag-app-sfdc-slack name=rag-app-sfdc-slack

# 2. Create secrets (use exported values)
kubectl create secret generic postgres-secret \
  --from-literal=postgres-password=YOUR_PASSWORD \
  --from-literal=password=YOUR_PASSWORD \
  -n rag-app-sfdc-slack

kubectl create secret generic sfdc-credentials \
  --from-literal=sessionId=YOUR_SESSION_ID \
  -n rag-app-sfdc-slack

kubectl create secret generic slack-credentials \
  --from-literal=botToken=xoxb-YOUR_TOKEN \
  --from-literal=signingSecret=YOUR_SECRET \
  -n rag-app-sfdc-slack

# 3. Verify
kubectl get secrets -n rag-app-sfdc-slack
```

### Deploy (5 min)
```bash
# Deploy using merged values
helm install rag-app-slack ./helm/rag-app-slack \
  -f helm/values-pcai-merged.yaml \
  -n rag-app-sfdc-slack

# Watch deployment
kubectl get pods -n rag-app-sfdc-slack -w

# Wait for ready
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/instance=rag-app-slack \
  -n rag-app-sfdc-slack \
  --timeout=300s
```

### Verify (5 min)
```bash
# Check pods
kubectl get pods -n rag-app-sfdc-slack

# Check logs
kubectl logs -n rag-app-sfdc-slack \
  -l app.kubernetes.io/instance=rag-app-slack

# Port-forward test
kubectl port-forward -n rag-app-sfdc-slack \
  svc/rag-app-slack 5001:5001 &

# Test endpoints
curl http://localhost:5001/health
curl http://localhost:5001/api/sfdc/status

# Stop port-forward
kill %1
```

### Cleanup (5 min)
```bash
# Remove old deployment (only if new version stable for 1 hour)
kubectl delete deployment rag-app -n rag-app

# Optional: Delete old namespace
kubectl delete namespace rag-app
```

---

## 🔴 CRITICAL SECURITY FIXES

### Before Migration
- [ ] Export SFDC session ID: `kubectl get pod -o yaml | grep -i session`
- [ ] Export DB password: `kubectl get pod -o yaml | grep DB_PASS`
- [ ] Backup existing: `kubectl get deployment rag-app -o yaml > backup.yaml`

### After Migration
- [ ] Credentials in K8s secrets (not env vars)
- [ ] Non-root user (1000 instead of 0)
- [ ] Privilege escalation disabled
- [ ] Network policies enabled
- [ ] Secrets not in git (only references)

---

## ⚠️ KEY DIFFERENCES TO KNOW

| Old | New | Action |
|-----|-----|--------|
| Port 80 | Port 5001 | Update DNS/ingress |
| 1 replica | 2 replicas | Auto-scaling enabled |
| Hardcoded credentials | K8s secrets | Create secrets first |
| Root user | Non-root (1000) | Container must support |
| Registry: sabya610/ | registry.pcai.local/ | Verify access |

---

## 🆘 TROUBLESHOOTING

### Pods not starting
```bash
kubectl logs rag-app-slack-0 -n rag-app-sfdc-slack
# Check: Secret not found, DB connection failed, model files missing
```

### Port 5001 not accessible
```bash
kubectl get ingress -n rag-app-sfdc-slack
kubectl port-forward svc/rag-app-slack 5001:5001
curl http://localhost:5001/health
```

### Database issues
```bash
kubectl exec postgres-slack-0 -n rag-app-sfdc-slack \
  -- psql -U postgres ragdb_slack -c "SELECT version();"
```

### SFDC not working
```bash
kubectl exec rag-app-slack-0 -n rag-app-sfdc-slack \
  -- cat /etc/sfdc/sid.txt
# If empty or expired, get new session ID
```

---

## 📊 EXPECTED RESULTS

### Pod Status
```
NAME                       READY   STATUS    RESTARTS
rag-app-slack-0           1/1     Running   0
rag-app-slack-1           1/1     Running   0  
postgres-slack-0          1/1     Running   0
```

### Endpoints Working
```
✅ GET  /health               → 200 OK
✅ GET  /api/slack/channels   → 200 + channels list
✅ GET  /api/sfdc/status      → 200 + SFDC status
✅ POST /api/slack/import     → 202 Accepted
```

### Resources
```
Memory:    3-4Gi per pod (requests 6Gi)
CPU:       300-500m per pod (requests 2000m)
Replicas:  2 (will scale to 3-5 under load)
```

---

## 📁 KEY FILES

| File | Use |
|------|-----|
| `helm/values-pcai-merged.yaml` | Deploy configuration |
| `MIGRATION_PLAYBOOK.md` | Detailed step-by-step |
| `MIGRATION_SUMMARY.md` | Comparison & recommendations |
| `VALUES_YAML_AUDIT.md` | Issue details |

---

## 🎯 SUCCESS CHECKLIST

- [ ] Pods running and healthy
- [ ] Health endpoint responding (200)
- [ ] SFDC integration working
- [ ] Slack integration working (if configured)
- [ ] Database accessible
- [ ] Autoscaling working
- [ ] Ingress accessible
- [ ] No errors in logs
- [ ] Resource usage normal
- [ ] DNS/monitoring updated

---

## ⏱️ TIMELINE

| Phase | Duration | What Happens |
|-------|----------|--------------|
| Prerequisites | 5 min | Verify cluster/registry/resources |
| Preparation | 5 min | Create namespace/secrets |
| Deployment | 5 min | Helm install and pod startup |
| Verification | 5 min | Health checks and testing |
| Cleanup | 5 min | Remove old deployment (optional) |
| **TOTAL** | **~30 min** | **Done!** |

---

## 📞 ESCALATION

If blocked:
1. Check logs: `kubectl logs rag-app-slack-0 -n rag-app-sfdc-slack`
2. Check secrets: `kubectl get secrets -n rag-app-sfdc-slack`
3. Verify registry: `kubectl describe pod rag-app-slack-0 -n rag-app-sfdc-slack`
4. Rollback if needed: `kubectl delete deployment rag-app-slack -n rag-app-sfdc-slack`

---

## ✅ Final Verification Command

```bash
# Run this after deployment to verify everything
echo "=== Pod Status ===" && \
kubectl get pods -n rag-app-sfdc-slack && \
echo "" && \
echo "=== Services ===" && \
kubectl get svc -n rag-app-sfdc-slack && \
echo "" && \
echo "=== PVCs ===" && \
kubectl get pvc -n rag-app-sfdc-slack && \
echo "" && \
echo "=== Health Check ===" && \
kubectl port-forward -n rag-app-sfdc-slack svc/rag-app-slack 5001:5001 & \
sleep 2 && \
curl http://localhost:5001/health && \
kill %1
```

**Expected Output:** ✅ All running, ✅ All bound, ✅ 200 OK

---

**Created:** 2026-06-26  
**Version:** 1.0  
**Status:** Ready for Migration
