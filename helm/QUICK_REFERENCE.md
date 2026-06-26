# Helm Quick Reference - RAG App Slack Integration

## One-Liner Deployment (Fastest)

```powershell
# Windows PowerShell
cd c:\Users\malliks\rag_app
.\helm\Deploy-Helm.ps1 -Action deploy
```

```bash
# Linux/Mac/WSL
cd /path/to/rag_app
chmod +x helm/deploy-helm.sh
./helm/deploy-helm.sh deploy
```

## Prerequisites Checklist

- [ ] Helm 3.x installed: `helm version`
- [ ] kubectl installed: `kubectl version`
- [ ] Cluster access: `kubectl cluster-info`
- [ ] Slack Bot Token (xoxb-...)
- [ ] Slack Signing Secret (32+ chars)
- [ ] SFDC credentials (optional)

## Manual Deployment

```bash
# Step 1: Add Helm repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Step 2: Lint chart
helm lint ./helm/rag-app-slack

# Step 3: Deploy
helm install rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set slack.botToken=xoxb-YOUR_TOKEN \
  --set slack.signingSecret=YOUR_SECRET \
  --wait

# Step 4: Verify
helm status rag-app-slack
kubectl get pods -l app.kubernetes.io/instance=rag-app-slack
```

## Verify Deployment

```bash
# Check status
helm status rag-app-slack -n default

# Check pods
kubectl get pods -n default -l app.kubernetes.io/instance=rag-app-slack

# View logs
kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack -f

# Port-forward and test
kubectl port-forward -n default svc/rag-app-slack 5001:5001
curl http://localhost:5001/api/slack/stats
```

## Quick Commands

```bash
# Get service info
kubectl get svc rag-app-slack

# Get service IP and port
kubectl get svc rag-app-slack -o jsonpath='{.spec.clusterIP}:{.spec.ports[0].port}'

# Scale deployment
kubectl scale deployment rag-app-slack --replicas=3

# Execute command in pod
kubectl exec -it POD_NAME -- curl http://localhost:5001/api/slack/stats

# Restart deployment
kubectl rollout restart deployment/rag-app-slack

# View rollout history
kubectl rollout history deployment/rag-app-slack
```

## Update Deployment

```bash
# Update with new values
helm upgrade rag-app-slack ./helm/rag-app-slack \
  -f ./helm/values-ftc-aie.yaml \
  --set slack.botToken=xoxb-NEW_TOKEN

# Verify update
helm status rag-app-slack
kubectl rollout status deployment/rag-app-slack
```

## Rollback

```bash
# View history
helm history rag-app-slack

# Rollback to previous
helm rollback rag-app-slack

# Rollback to specific revision
helm rollback rag-app-slack 1
```

## Uninstall

```bash
# Uninstall (keeps data)
helm uninstall rag-app-slack

# Delete with data
helm uninstall rag-app-slack
kubectl delete pvc -l app.kubernetes.io/instance=rag-app-slack
```

## Troubleshooting

```bash
# Pod status
kubectl describe pod POD_NAME

# Pod logs
kubectl logs POD_NAME -f

# Recent events
kubectl get events --sort-by='.lastTimestamp'

# Resource usage
kubectl top pods

# Check database
kubectl exec -it postgres-slack-0 -- psql -U postgres -d ragdb_slack -c "\dt"
```

## API Testing

```bash
# Get stats
curl http://localhost:5001/api/slack/stats

# Get channels
curl http://localhost:5001/api/slack/channels

# Import messages
curl -X POST http://localhost:5001/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{"channel_ids":["C123456"],"days_back":7}'

# Search messages
curl -X POST http://localhost:5001/api/slack/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","search_type":"semantic"}'

# Get thread
curl http://localhost:5001/api/slack/threads/ts123456
```

## Files Reference

| File | Purpose |
|------|---------|
| `helm/rag-app-slack/Chart.yaml` | Chart metadata |
| `helm/rag-app-slack/values.yaml` | Default values |
| `helm/values-ftc-aie.yaml` | FTC cluster values |
| `helm/rag-app-slack/templates/deployment.yaml` | K8s deployment |
| `helm/deploy-helm.sh` | Bash deployment script |
| `helm/Deploy-Helm.ps1` | PowerShell deployment script |
| `helm/HELM_DEPLOYMENT_GUIDE.md` | Full documentation |

---

**Quick Help:** `helm status rag-app-slack`  
**Full Docs:** See `helm/HELM_DEPLOYMENT_GUIDE.md`
