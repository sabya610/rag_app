# START HERE - Quick Deployment Guide

## Step 1: Prerequisites (5 minutes)

Get your Slack credentials from https://api.slack.com/apps

- Bot Token - Copy the value starting with xoxb-
- Signing Secret - 32+ character string

Keep these safe!

## Step 2: Choose Your Deployment Method

### Option A: Automated (Recommended)

**Windows PowerShell:**

cd helm
.\Deploy-Helm.ps1 -Action deploy

**Linux/Mac/WSL:**

cd helm
chmod +x deploy-helm.sh
./deploy-helm.sh deploy

## Step 3: Verify Deployment

# Check pods
kubectl get pods -n default -l app.kubernetes.io/instance=rag-app-slack

# View logs
kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack -f

## Troubleshooting

**Pods in CrashLoopBackOff?**
kubectl logs -n default POD_NAME -f

**Cannot connect to database?**
kubectl get statefulset -n default
kubectl logs -n default postgres-slack-0

## Next: Documentation

- HELM_PACKAGE_SUMMARY.md - Full overview
- DEPLOYMENT_CHECKLIST.md - Pre-deployment check
- helm/HELM_DEPLOYMENT_GUIDE.md - Complete guide
