param(
    [ValidateSet('create', 'verify', 'upload')]
    [string]$Action = 'create'
)

$releaseDir = "./releases"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$packageName = "rag-app-slack-helm-deployment_$timestamp"
$packagePath = Join-Path $releaseDir $packageName

Write-Host "`n" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "RAG APP SLACK INTEGRATION - HELM PACKAGE CREATOR" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "`n"

# Create package directory structure
Write-Host "Creating package structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$packagePath/helm" -Force | Out-Null

# Copy Helm chart
Write-Host "Copying Helm chart..." -ForegroundColor Yellow
Copy-Item -Path "./helm/rag-app-slack" -Destination "$packagePath/helm/" -Recurse -Force

# Copy configuration files
Write-Host "Copying configuration files..." -ForegroundColor Yellow
Copy-Item -Path "./helm/values-ftc-aie.yaml" -Destination "$packagePath/helm/" -Force
Copy-Item -Path "./helm/HELM_DEPLOYMENT_GUIDE.md" -Destination "$packagePath/helm/" -Force
Copy-Item -Path "./helm/QUICK_REFERENCE.md" -Destination "$packagePath/helm/" -Force

# Copy deployment scripts
Write-Host "Copying deployment scripts..." -ForegroundColor Yellow
Copy-Item -Path "./helm/Deploy-Helm.ps1" -Destination "$packagePath/helm/" -Force
Copy-Item -Path "./helm/deploy-helm.sh" -Destination "$packagePath/helm/" -Force

# Copy documentation
Write-Host "Copying documentation..." -ForegroundColor Yellow
Copy-Item -Path "./DEPLOYMENT_CHECKLIST.md" -Destination "$packagePath/" -Force
Copy-Item -Path "./HELM_PACKAGE_SUMMARY.md" -Destination "$packagePath/" -Force

# Create README file
$readmeLines = @(
    "# RAG App Slack Integration - Helm Deployment Package"
    ""
    "Status: Ready for Production Deployment"
    ""
    "## Package Contents"
    ""
    "- helm/rag-app-slack/ - Complete Helm chart"
    "- helm/values-ftc-aie.yaml - FTC AIE cluster configuration"
    "- helm/Deploy-Helm.ps1 - Automated deployment (Windows)"
    "- helm/deploy-helm.sh - Automated deployment (Linux/Mac/WSL)"
    "- helm/HELM_DEPLOYMENT_GUIDE.md - Complete guide (20+ pages)"
    "- helm/QUICK_REFERENCE.md - Quick commands"
    "- DEPLOYMENT_CHECKLIST.md - Pre-deployment verification"
    "- HELM_PACKAGE_SUMMARY.md - Full overview"
    ""
    "## Quick Start"
    ""
    "### Windows PowerShell"
    ""
    "cd helm"
    ".\Deploy-Helm.ps1 -Action deploy"
    ""
    "### Linux/Mac/WSL"
    ""
    "cd helm"
    "chmod +x deploy-helm.sh"
    "./deploy-helm.sh deploy"
    ""
    "## Documentation"
    ""
    "Read in this order:"
    "1. README.md (start here)"
    "2. HELM_PACKAGE_SUMMARY.md (overview)"
    "3. DEPLOYMENT_CHECKLIST.md (requirements)"
    "4. helm/HELM_DEPLOYMENT_GUIDE.md (detailed guide)"
    "5. helm/QUICK_REFERENCE.md (commands)"
    ""
    "## Prerequisites"
    ""
    "- Helm 3.x installed"
    "- kubectl configured for your cluster"
    "- Slack Bot Token (from https://api.slack.com/apps)"
    "- Slack Signing Secret"
    "- SFDC credentials (optional)"
    ""
    "## Support"
    ""
    "- GitHub: https://github.com/sabya610/rag_app"
    "- Helm Docs: https://helm.sh/docs/"
    "- Kubernetes: https://kubernetes.io/docs/"
)

Set-Content -Path "$packagePath/README.md" -Value $readmeLines

# Create START_HERE file
$startHereLines = @(
    "# START HERE - Quick Deployment Guide"
    ""
    "## Step 1: Prerequisites (5 minutes)"
    ""
    "Get your Slack credentials from https://api.slack.com/apps"
    ""
    "- Bot Token - Copy the value starting with xoxb-"
    "- Signing Secret - 32+ character string"
    ""
    "Keep these safe!"
    ""
    "## Step 2: Choose Your Deployment Method"
    ""
    "### Option A: Automated (Recommended)"
    ""
    "**Windows PowerShell:**"
    ""
    "cd helm"
    ".\Deploy-Helm.ps1 -Action deploy"
    ""
    "**Linux/Mac/WSL:**"
    ""
    "cd helm"
    "chmod +x deploy-helm.sh"
    "./deploy-helm.sh deploy"
    ""
    "## Step 3: Verify Deployment"
    ""
    "# Check pods"
    "kubectl get pods -n default -l app.kubernetes.io/instance=rag-app-slack"
    ""
    "# View logs"
    "kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack -f"
    ""
    "## Troubleshooting"
    ""
    "**Pods in CrashLoopBackOff?**"
    "kubectl logs -n default POD_NAME -f"
    ""
    "**Cannot connect to database?**"
    "kubectl get statefulset -n default"
    "kubectl logs -n default postgres-slack-0"
    ""
    "## Next: Documentation"
    ""
    "- HELM_PACKAGE_SUMMARY.md - Full overview"
    "- DEPLOYMENT_CHECKLIST.md - Pre-deployment check"
    "- helm/HELM_DEPLOYMENT_GUIDE.md - Complete guide"
)

Set-Content -Path "$packagePath/START_HERE.md" -Value $startHereLines

# Create archive
Write-Host "Creating compressed archive..." -ForegroundColor Yellow
$zipPath = Join-Path $releaseDir ($packageName + ".zip")
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Compress-Archive -Path $packagePath -DestinationPath $zipPath -Force

# Display results
Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "HELM DEPLOYMENT PACKAGE CREATED SUCCESSFULLY" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "`n"

$packageFile = Get-Item $zipPath
Write-Host "Package Details:" -ForegroundColor Green
Write-Host "  Name:     $($packageFile.Name)" -ForegroundColor Yellow
Write-Host "  Size:     $([math]::Round($packageFile.Length / 1MB, 2)) MB" -ForegroundColor Yellow
Write-Host "  Location: $($packageFile.FullName)" -ForegroundColor Yellow

Write-Host "`nPackage Contents:" -ForegroundColor Green
Write-Host "  - Complete Helm chart" -ForegroundColor Cyan
Write-Host "  - Deployment automation scripts" -ForegroundColor Cyan
Write-Host "  - FTC cluster configuration" -ForegroundColor Cyan
Write-Host "  - Comprehensive documentation" -ForegroundColor Cyan
Write-Host "  - Pre-deployment checklist" -ForegroundColor Cyan

Write-Host "`nNext Steps:" -ForegroundColor Green
Write-Host "  1. Extract the ZIP file" -ForegroundColor Cyan
Write-Host "  2. Open START_HERE.md" -ForegroundColor Cyan
Write-Host "  3. Follow the deployment guide" -ForegroundColor Cyan

Write-Host "`n════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan
