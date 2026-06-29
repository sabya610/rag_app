# RAG App Slack Integration - Helm Deployment Package

Status: Ready for Production Deployment

## Package Contents

- helm/rag-app-slack/ - Complete Helm chart
- helm/values-ftc-aie.yaml - FTC AIE cluster configuration
- helm/Deploy-Helm.ps1 - Automated deployment (Windows)
- helm/deploy-helm.sh - Automated deployment (Linux/Mac/WSL)
- helm/HELM_DEPLOYMENT_GUIDE.md - Complete guide (20+ pages)
- helm/QUICK_REFERENCE.md - Quick commands
- DEPLOYMENT_CHECKLIST.md - Pre-deployment verification
- HELM_PACKAGE_SUMMARY.md - Full overview

## Quick Start

### Windows PowerShell

cd helm
.\Deploy-Helm.ps1 -Action deploy

### Linux/Mac/WSL

cd helm
chmod +x deploy-helm.sh
./deploy-helm.sh deploy

## Documentation

Read in this order:
1. README.md (start here)
2. HELM_PACKAGE_SUMMARY.md (overview)
3. DEPLOYMENT_CHECKLIST.md (requirements)
4. helm/HELM_DEPLOYMENT_GUIDE.md (detailed guide)
5. helm/QUICK_REFERENCE.md (commands)

## Prerequisites

- Helm 3.x installed
- kubectl configured for your cluster
- Slack Bot Token (from https://api.slack.com/apps)
- Slack Signing Secret
- SFDC credentials (optional)

## Support

- GitHub: https://github.com/sabya610/rag_app
- Helm Docs: https://helm.sh/docs/
- Kubernetes: https://kubernetes.io/docs/
