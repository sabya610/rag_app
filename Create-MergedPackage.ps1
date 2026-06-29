param(
    [string]$OutputFile = "rag-app-slack-merged-deployment.tar.gz"
)

$releaseDir = "./releases"
$tempDir = "./temp_merged_package"
$packageName = "rag-app-slack-merged-deployment"

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "CREATING MERGED DEPLOYMENT PACKAGE WITH PCAI VALUES" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "`n"

# Cleanup
if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
New-Item -ItemType Directory -Path "$tempDir/$packageName" -Force | Out-Null

Write-Host "Building package structure..." -ForegroundColor Yellow

# Create helm directory
New-Item -ItemType Directory -Path "$tempDir/$packageName/helm" -Force | Out-Null

# Copy Helm chart
Write-Host "  - Copying Helm chart..." -ForegroundColor Cyan
Copy-Item -Path "./helm/rag-app-slack" -Destination "$tempDir/$packageName/helm/" -Recurse -Force

# Copy ALL values files
Write-Host "  - Copying values files..." -ForegroundColor Cyan
Copy-Item -Path "./helm/values-ftc-aie.yaml" -Destination "$tempDir/$packageName/helm/" -Force
Copy-Item -Path "./helm/values-pcai.yaml" -Destination "$tempDir/$packageName/helm/" -Force
Copy-Item -Path "./helm/values-pcai-merged.yaml" -Destination "$tempDir/$packageName/helm/" -Force

# Copy deployment guides
Write-Host "  - Copying deployment guides..." -ForegroundColor Cyan
Copy-Item -Path "./helm/HELM_DEPLOYMENT_GUIDE.md" -Destination "$tempDir/$packageName/helm/" -Force
Copy-Item -Path "./helm/QUICK_REFERENCE.md" -Destination "$tempDir/$packageName/helm/" -Force
Copy-Item -Path "./helm/Deploy-Helm.ps1" -Destination "$tempDir/$packageName/helm/" -Force
Copy-Item -Path "./helm/deploy-helm.sh" -Destination "$tempDir/$packageName/helm/" -Force

# Copy root documentation
Write-Host "  - Copying documentation..." -ForegroundColor Cyan
Copy-Item -Path "./DEPLOYMENT_CHECKLIST.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./HELM_PACKAGE_SUMMARY.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./MIGRATION_INDEX.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./MIGRATION_SUMMARY.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./MIGRATION_COMPARISON.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./MIGRATION_PLAYBOOK.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./QUICK_MIGRATION_CARD.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./VALUES_YAML_AUDIT.md" -Destination "$tempDir/$packageName/" -Force
Copy-Item -Path "./PCAI_DEPLOYMENT_GUIDE.md" -Destination "$tempDir/$packageName/" -Force

# Create deployment README
$readmeContent = @(
    "# RAG App Slack Integration - PCAI Merged Deployment Package"
    ""
    "Status: PRODUCTION-READY"
    ""
    "## Quick Start"
    ""
    "### Extract Package"
    "tar -xzf rag-app-slack-merged-deployment.tar.gz"
    "cd rag-app-slack-merged-deployment"
    ""
    "### Deploy to PCAI"
    ""
    "#### Windows PowerShell"
    "cd helm"
    ".\Deploy-Helm.ps1"
    ""
    "#### Linux/Mac/WSL"
    "cd helm"
    "chmod +x deploy-helm.sh"
    "./deploy-helm.sh"
    ""
    "### Key Files"
    ""
    "- helm/values-pcai-merged.yaml .... RECOMMENDED for migration"
    "- helm/values-pcai.yaml ........... Fresh PCAI install"
    "- helm/values-ftc-aie.yaml ........ FTC cluster specific"
    ""
    "### Documentation Order"
    ""
    "1. MIGRATION_INDEX.md ............. START HERE"
    "2. QUICK_MIGRATION_CARD.md ........ 30-minute fast track"
    "3. MIGRATION_SUMMARY.md ........... Comparison overview"
    "4. MIGRATION_PLAYBOOK.md .......... Step-by-step guide"
    "5. helm/HELM_DEPLOYMENT_GUIDE.md .. Complete deployment reference"
    ""
    "## What's Included"
    ""
    "✅ Complete Helm chart"
    "✅ Three values files (FTC, PCAI basic, PCAI merged)"
    "✅ Deployment automation scripts"
    "✅ Comprehensive documentation (9 guides)"
    "✅ Security audit and fixes"
    "✅ Migration playbooks and checklists"
    "✅ Quick reference cards"
    ""
    "## Configuration Files"
    ""
    "helm/values-pcai-merged.yaml - RECOMMENDED"
    "  - Merges existing deployment + PCAI improvements"
    "  - Preserves SFDC and EZUA integration"
    "  - Adds Slack integration"
    "  - Implements all security fixes"
    "  - Ready for production migration"
    ""
    "helm/values-pcai.yaml"
    "  - Basic PCAI configuration"
    "  - Use for fresh installations"
    "  - Includes all security hardening"
    ""
    "helm/values-ftc-aie.yaml"
    "  - Optimized for FTC AIE 1.1.1 cluster"
    "  - Pre-configured with cluster settings"
    ""
    "## Support"
    ""
    "See MIGRATION_INDEX.md for documentation navigation"
    "See QUICK_MIGRATION_CARD.md for deployment commands"
    "See MIGRATION_PLAYBOOK.md for troubleshooting"
)

Set-Content -Path "$tempDir/$packageName/README.md" -Value $readmeContent

# Create deployment info file
$infoContent = @(
    "RAG APP SLACK INTEGRATION - HELM DEPLOYMENT PACKAGE"
    ""
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "Version: 1.0"
    "Status: Production Ready"
    ""
    "PACKAGE CONTENTS:"
    "- helm/rag-app-slack/ .............. Complete Helm chart"
    "- helm/values-pcai-merged.yaml .... RECOMMENDED values file"
    "- helm/values-pcai.yaml ........... PCAI basic configuration"
    "- helm/values-ftc-aie.yaml ........ FTC cluster configuration"
    "- helm/Deploy-Helm.ps1 ............ Windows deployment script"
    "- helm/deploy-helm.sh ............. Linux/Mac deployment script"
    ""
    "DOCUMENTATION:"
    "- MIGRATION_INDEX.md .............. Navigation guide (START HERE)"
    "- QUICK_MIGRATION_CARD.md ......... 30-minute deployment guide"
    "- MIGRATION_SUMMARY.md ............ Comparison and recommendations"
    "- MIGRATION_COMPARISON.md ......... Detailed technical comparison"
    "- MIGRATION_PLAYBOOK.md ........... Step-by-step migration procedures"
    "- VALUES_YAML_AUDIT.md ............ Security audit results"
    "- PCAI_DEPLOYMENT_GUIDE.md ........ PCAI-specific setup"
    "- helm/HELM_DEPLOYMENT_GUIDE.md ... Complete deployment reference"
    ""
    "KEY FILES FOR DOWNLOAD:"
    "1. helm/values-pcai-merged.yaml ... Your deployment configuration"
    "2. This tar.gz package ............ All files for deployment"
    ""
    "RECOMMENDED DEPLOYMENT:"
    "helm install rag-app-slack ./helm/rag-app-slack \\"
    "  -f helm/values-pcai-merged.yaml \\"
    "  -n rag-app-sfdc-slack"
    ""
    "SECURITY IMPROVEMENTS:"
    "✅ Credentials moved to Kubernetes Secrets"
    "✅ Running as non-root user (1000)"
    "✅ Privilege escalation disabled"
    "✅ Network policies enabled"
    "✅ Pod security context hardened"
    ""
    "NEW FEATURES:"
    "✅ Slack integration"
    "✅ HA with autoscaling (2-5 pods)"
    "✅ Persistent model storage"
    "✅ Production ingress configuration"
    "✅ Enhanced monitoring"
)

Set-Content -Path "$tempDir/$packageName/DEPLOYMENT_INFO.txt" -Value $infoContent

# Create archive
Write-Host "Creating tar.gz archive..." -ForegroundColor Yellow
$fullTarPath = (Resolve-Path $releaseDir).Path + "\" + $OutputFile
Push-Location $tempDir
tar -czf $fullTarPath $packageName
Pop-Location

# Cleanup
Remove-Item $tempDir -Recurse -Force

# Display results
Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "DEPLOYMENT PACKAGE CREATED SUCCESSFULLY" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green

$file = Get-Item $fullTarPath
Write-Host "`nPackage Details:" -ForegroundColor Green
Write-Host "  Name:     $($file.Name)" -ForegroundColor Yellow
Write-Host "  Size:     $([math]::Round($file.Length / 1MB, 2)) MB" -ForegroundColor Yellow
Write-Host "  Location: $($file.FullName)" -ForegroundColor Yellow
Write-Host "  Created:  $($file.LastWriteTime)" -ForegroundColor Yellow

Write-Host "`nPackage Contents:" -ForegroundColor Green
Write-Host "  - helm/rag-app-slack/ (complete Helm chart)" -ForegroundColor Cyan
Write-Host "  - helm/values-pcai-merged.yaml (RECOMMENDED)" -ForegroundColor Cyan
Write-Host "  - helm/values-pcai.yaml (alternative)" -ForegroundColor Cyan
Write-Host "  - helm/values-ftc-aie.yaml (FTC cluster)" -ForegroundColor Cyan
Write-Host "  - Deployment scripts and guides" -ForegroundColor Cyan
Write-Host "  - 9 comprehensive documentation files" -ForegroundColor Cyan
Write-Host "  - Migration playbooks and checklists" -ForegroundColor Cyan

Write-Host "`nKey File for Your Deployment:" -ForegroundColor Green
Write-Host "  📄 helm/values-pcai-merged.yaml" -ForegroundColor Yellow
Write-Host "     (Configure with your secrets and use this for migration)" -ForegroundColor Cyan

Write-Host "`nHow to Download:" -ForegroundColor Green
Write-Host "  File: releases/$($file.Name)" -ForegroundColor Yellow
Write-Host "  URL: [See releases folder in your workspace]" -ForegroundColor Cyan

Write-Host "`nNext Steps:" -ForegroundColor Green
Write-Host "  1. Download the tar.gz file" -ForegroundColor Cyan
Write-Host "  2. Extract: tar -xzf $($file.Name)" -ForegroundColor Cyan
Write-Host "  3. Read: MIGRATION_INDEX.md" -ForegroundColor Cyan
Write-Host "  4. Deploy: Follow QUICK_MIGRATION_CARD.md" -ForegroundColor Cyan

Write-Host "`n════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan
