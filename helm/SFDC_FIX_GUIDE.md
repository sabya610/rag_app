# SFDC Session ID Configuration Fix - Deployment Guide

## Problem
The Helm deployment was not properly creating and mounting the SFDC session ID secret, causing the app to show "SFDC: not configured".

## Solution
Updated Helm templates to:
1. Create a Kubernetes Secret with SFDC credentials (sfdc-secret.yaml)
2. Mount the secret as a file at `/etc/sfdc/sid.txt` in the pod
3. Pass SFDC environment variables from the secret

## Files Updated
- `helm/rag-app/templates/sfdc-secret.yaml` - NEW: Creates Secret for SFDC credentials
- `helm/rag-app/templates/deployment.yaml` - UPDATED: Mounts SFDC secret and sets environment variables
- `helm/rag-app/values.yaml` - UPDATED: Added SFDC configuration section

## Steps to Redeploy

### 1. Verify Values File has Session ID
```bash
# Your values.yaml should have:
sfdc:
  sessionId: "00Dd0000000bUlK!ARAAQOK8_uF259fTsrTEOqnIwYwOlpORuawjCyjrt1P8IILRB05UVw_DbBxgq5XQ23dRPt1bBFuxGNSey.9hFbN44evL2QeL"
  sessionIdFile: "/etc/sfdc/sid.txt"
  clientId: ""
  clientSecret: ""
  username: ""
  password: ""
  securityToken: ""
  loginUrl: "https://login.salesforce.com"
```

### 2. Delete the old release
```bash
helm delete rag-app -n rag-app-sfdc-slack
# or your namespace
```

### 3. Reinstall with updated chart
```bash
# Extract the tar.gz
tar -xzf rag-app-helm.tar.gz

# Install with your values file
helm install rag-app ./rag-app -f values.yaml -n rag-app-sfdc-slack --create-namespace

# Or install with PCAI specific values
helm install rag-app ./rag-app -f helm/values-pcai-merged.yaml -n rag-app-sfdc-slack --create-namespace
```

### 4. Verify Secret Creation
```bash
# Check if the SFDC secret was created
kubectl get secret -n rag-app-sfdc-slack
# Output should show: rag-app-sfdc

# View the secret (it's base64 encoded)
kubectl describe secret rag-app-sfdc -n rag-app-sfdc-slack

# Verify the session ID file is mounted in the pod
kubectl exec -it <pod-name> -n rag-app-sfdc-slack -- cat /etc/sfdc/sid.txt
```

### 5. Check Pod Logs
```bash
kubectl logs -f <pod-name> -n rag-app-sfdc-slack

# Look for messages like:
# [SFDC] Authenticated successfully.
# or any SFDC connection errors
```

### 6. Test SFDC Connectivity
```bash
# Port-forward to test
kubectl port-forward svc/rag-app 5001:80 -n rag-app-sfdc-slack

# Visit http://localhost:5001 and check:
# - "SFDC: connected" (success)
# - "SFDC: not configured" (session ID not found or wrong)
# - "SFDC: disconnected" (session ID invalid or authentication failed)
```

## Environment Variables Set by Helm

The Helm chart now sets:
- `SFDC_ENABLED`: "true"
- `SF_URL`: "https://hp.my.salesforce.com"
- `SFDC_PRODUCT_QUEUE`: "HPE Ezmeral"
- `SFDC_PRODUCT_LINE`: "CONT PLT SW (RM)"
- `SFDC_LOGIN_URL`: "https://login.salesforce.com"
- `SFDC_CLIENT_ID`: (from secret if OAuth2 mode)
- `SFDC_CLIENT_SECRET`: (from secret if OAuth2 mode)
- `SFDC_USERNAME`: (from secret if OAuth2 mode)
- `SFDC_PASSWORD`: (from secret if OAuth2 mode)
- `SFDC_SECURITY_TOKEN`: (from secret if OAuth2 mode)

## File Mount
The session ID file is mounted at:
- **Inside pod**: `/etc/sfdc/sid.txt`
- **Values reference**: `sessionIdFile: "/etc/sfdc/sid.txt"`
- **Secret key**: `sid.txt` (from SFDC Secret)

## Troubleshooting

### "SFDC: not configured"
- Session ID might be empty or invalid
- Check: `kubectl describe secret rag-app-sfdc -n <namespace>`
- Verify: `kubectl exec -it <pod-name> -n <namespace> -- cat /etc/sfdc/sid.txt`

### "SFDC: disconnected"
- Session ID is being read but authentication failed
- Check Salesforce session ID expiration
- Verify Salesforce URL is correct in values.yaml

### Secret not created
- Check Helm templates syntax: `helm template ./rag-app -f values.yaml`
- Verify SFDC values are set in values.yaml
- Re-install: `helm install rag-app ./rag-app ...`

## Next Steps
1. Redeploy using the updated Helm chart
2. Verify secret creation with kubectl
3. Check pod logs for SFDC connection status
4. Access web UI and verify "SFDC: connected" status
