@echo off
REM RAG App Slack Integration - Helm Package Creator (Windows)
REM Creates a downloadable deployment package

setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     RAG APP SLACK INTEGRATION - HELM PACKAGE CREATOR         ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found
    exit /b 1
)

REM Run PowerShell script to create package
powershell -NoProfile -ExecutionPolicy Bypass -Command "& {
    $releaseDir = './releases'
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $packageName = 'rag-app-slack-helm-deployment_' + $timestamp
    $packagePath = Join-Path $releaseDir $packageName
    
    # Create package directory
    New-Item -ItemType Directory -Path $packagePath -Force | Out-Null
    New-Item -ItemType Directory -Path $packagePath/helm -Force | Out-Null
    
    # Copy Helm chart
    Copy-Item -Path './helm/rag-app-slack' -Destination $packagePath/helm/ -Recurse -Force
    
    # Copy configuration files
    Copy-Item -Path './helm/values-ftc-aie.yaml' -Destination $packagePath/helm/ -Force
    Copy-Item -Path './helm/HELM_DEPLOYMENT_GUIDE.md' -Destination $packagePath/helm/ -Force
    Copy-Item -Path './helm/QUICK_REFERENCE.md' -Destination $packagePath/helm/ -Force
    
    # Copy deployment scripts
    Copy-Item -Path './helm/Deploy-Helm.ps1' -Destination $packagePath/helm/ -Force
    Copy-Item -Path './helm/deploy-helm.sh' -Destination $packagePath/helm/ -Force
    
    # Copy checklists
    Copy-Item -Path './DEPLOYMENT_CHECKLIST.md' -Destination $packagePath/ -Force
    Copy-Item -Path './HELM_PACKAGE_SUMMARY.md' -Destination $packagePath/ -Force
    
    # Create comprehensive README
    $readme = @'
# RAG App Slack Integration - Helm Deployment Package

**Ready for Production Deployment**

## 📦 Package Contents

- `helm/rag-app-slack/` - Complete Helm chart (production-ready)
- `helm/values-ftc-aie.yaml` - FTC AIE cluster configuration
- `helm/Deploy-Helm.ps1` - Automated deployment (Windows PowerShell)
- `helm/deploy-helm.sh` - Automated deployment (Linux/Mac/WSL)
- `helm/HELM_DEPLOYMENT_GUIDE.md` - Complete deployment guide (20+ pages)
- `helm/QUICK_REFERENCE.md` - Quick commands & troubleshooting
- `DEPLOYMENT_CHECKLIST.md` - Pre-deployment verification
- `HELM_PACKAGE_SUMMARY.md` - Full overview

## 🚀 Quick Start

### Prerequisites
1. Helm 3.x installed
2. kubectl configured for your cluster
3. Slack Bot Token (from https://api.slack.com/apps)
4. Slack Signing Secret

### Deployment (Windows)

```powershell
# Navigate to package directory
cd rag-app-slack-helm-deployment_*

# Run deployment
.\helm\Deploy-Helm.ps1 -Action deploy

# When prompted, enter:
# - Slack Bot Token (xoxb-...)
# - Slack Signing Secret
# - SFDC credentials (optional)
```

### Deployment (Linux/Mac)

```bash
# Navigate to package directory
cd rag-app-slack-helm-deployment_*

# Make script executable
chmod +x helm/deploy-helm.sh

# Run deployment
./helm/deploy-helm.sh deploy
```

## 📚 Documentation

1. **Start with:** `HELM_PACKAGE_SUMMARY.md` (overview)
2. **Check requirements:** `DEPLOYMENT_CHECKLIST.md` (prerequisites)
3. **For details:** `helm/HELM_DEPLOYMENT_GUIDE.md` (20+ pages)
4. **Quick commands:** `helm/QUICK_REFERENCE.md` (reference)

## ✅ Verify Deployment

After deployment, verify using:

```bash
# Check pods
kubectl get pods -n default -l app.kubernetes.io/instance=rag-app-slack

# Check logs
kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack -f

# Test endpoint
kubectl port-forward -n default svc/rag-app-slack 5001:5001
curl http://localhost:5001/api/slack/stats
```

## 🎯 Cluster Information

**Target Cluster:** FTC AIE 1.1.1  
**Deployment Type:** Kubernetes with Helm  
**Replicas:** 2 (auto-scales 2-4)  
**Database:** PostgreSQL (100GB)  
**Memory:** 3-6GB per pod  
**CPU:** 1.5-3 per pod

## 📞 Support

- Full Guide: `helm/HELM_DEPLOYMENT_GUIDE.md`
- GitHub: https://github.com/sabya610/rag_app
- Helm Docs: https://helm.sh/docs/
- Kubernetes: https://kubernetes.io/docs/

## 📋 Included in Package

✅ Production-ready Helm chart  
✅ Automated deployment scripts  
✅ FTC cluster optimization  
✅ Complete documentation  
✅ Pre-deployment checklist  
✅ Quick reference guide  
✅ Troubleshooting help

## 🎉 Next Steps

1. Extract this package
2. Read `HELM_PACKAGE_SUMMARY.md`
3. Check `DEPLOYMENT_CHECKLIST.md`
4. Get Slack credentials
5. Run deployment script
6. Verify with test commands

---

**Generated:** {0}
**Version:** 1.0.0
**Repository:** https://github.com/sabya610/rag_app

---

For detailed instructions, open `helm/HELM_DEPLOYMENT_GUIDE.md`
'@ -f (Get-Date)
    
    Set-Content -Path $packagePath/README.md -Value $readme -Force
    
    # Create deployment guide quick link
    $quickStart = @'
# QUICK START

## 1. Windows PowerShell Deployment

```powershell
cd helm
.\Deploy-Helm.ps1 -Action deploy
```

## 2. Linux/Mac Deployment

```bash
cd helm
chmod +x deploy-helm.sh
./deploy-helm.sh deploy
```

## 3. Manual Deployment

```bash
helm install rag-app-slack ./rag-app-slack \
  -f values-ftc-aie.yaml \
  --set slack.botToken=xoxb-YOUR_TOKEN \
  --set slack.signingSecret=YOUR_SECRET
```

## Documentation Files

Open these in order:
1. README.md (this directory)
2. HELM_PACKAGE_SUMMARY.md
3. DEPLOYMENT_CHECKLIST.md  
4. helm/HELM_DEPLOYMENT_GUIDE.md

See helm/QUICK_REFERENCE.md for commands
'@
    
    Set-Content -Path $packagePath/START_HERE.md -Value $quickStart -Force
    
    # Create zip file
    $zipPath = Join-Path $releaseDir ($packageName + '.zip')
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    
    # Use built-in compression
    Compress-Archive -Path $packagePath -DestinationPath $zipPath -Force
    
    # Display results
    Write-Host "`n================================================================================`n" -ForegroundColor Cyan
    Write-Host 'HELM DEPLOYMENT PACKAGE CREATED' -ForegroundColor Green -NoNewline
    Write-Host '    ✅' -ForegroundColor Green
    Write-Host '================================================================================`n' -ForegroundColor Cyan
    
    Get-Item $zipPath | ForEach-Object {
        Write-Host 'Package Name:      ' -NoNewline
        Write-Host $_.Name -ForegroundColor Yellow
        
        Write-Host 'Size:              ' -NoNewline
        Write-Host ([math]::Round($_.Length / 1MB, 2)).ToString() + ' MB' -ForegroundColor Yellow
        
        Write-Host 'Location:          ' -NoNewline
        Write-Host $_.FullName -ForegroundColor Yellow
        
        Write-Host 'Created:           ' -NoNewline
        Write-Host $_.LastWriteTime -ForegroundColor Yellow
    }
    
    Write-Host '================================================================================`n' -ForegroundColor Cyan
    Write-Host 'NEXT STEPS:' -ForegroundColor Green
    Write-Host '1. Download the ZIP file above' -ForegroundColor Cyan
    Write-Host '2. Extract to your desired location' -ForegroundColor Cyan
    Write-Host '3. Open README.md in the extracted folder' -ForegroundColor Cyan
    Write-Host '4. Follow instructions for deployment' -ForegroundColor Cyan
    Write-Host '================================================================================`n' -ForegroundColor Cyan
}
"

echo.
echo Package creation complete! Check the releases directory.
echo.
