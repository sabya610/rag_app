# PowerShell Helm Chart Packaging Script

param(
    [string]$HelmChartDir = "./helm/rag-app-slack",
    [string]$OutputDir = "./helm-releases"
)

# Color Functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Cyan
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Main Logic
Write-Header "Creating Helm Package for Distribution"

# Verify Helm is available
if (-not (Get-Command helm -ErrorAction SilentlyContinue)) {
    Write-Warning "Helm not found. Please install Helm 3.x"
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Success "Output directory created"
}

# Package Helm chart
Write-Info "Packaging Helm chart..."
helm package $HelmChartDir --destination $OutputDir

# Get the package file
$packages = Get-ChildItem -Path "$OutputDir/rag-app-slack-*.tgz" | Sort-Object LastWriteTime -Descending
if ($packages) {
    $package = $packages[0]
    Write-Success "Chart packaged: $($package.Name)"
    
    # Create deployment bundle
    Write-Info "Creating deployment bundle..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $bundleDir = "rag-app-slack-deployment-$timestamp"
    
    New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null
    
    # Copy files
    Copy-Item $package.FullName "$bundleDir/" -Force
    Copy-Item ".\helm\values-ftc-aie.yaml" "$bundleDir\" -ErrorAction SilentlyContinue
    Copy-Item ".\helm\deploy-helm.sh" "$bundleDir\" -ErrorAction SilentlyContinue
    Copy-Item ".\helm\Deploy-Helm.ps1" "$bundleDir\" -ErrorAction SilentlyContinue
    Copy-Item ".\helm\HELM_DEPLOYMENT_GUIDE.md" "$bundleDir\" -ErrorAction SilentlyContinue
    Copy-Item ".\helm\QUICK_REFERENCE.md" "$bundleDir\" -ErrorAction SilentlyContinue
    
    # Create README
    $readmeContent = @"
# RAG App Slack Integration - Helm Deployment Package

This package contains everything needed to deploy the RAG App with Slack Integration to a Kubernetes cluster.

## Contents

- ``rag-app-slack-*.tgz`` - Helm chart package
- ``values-ftc-aie.yaml`` - FTC AIE 1.1.1 cluster values
- ``deploy-helm.sh`` - Bash deployment script
- ``Deploy-Helm.ps1`` - PowerShell deployment script
- ``HELM_DEPLOYMENT_GUIDE.md`` - Comprehensive deployment guide
- ``QUICK_REFERENCE.md`` - Quick reference commands

## Quick Start

### Windows PowerShell
```powershell
.\Deploy-Helm.ps1 -Action deploy
```

### Linux/Mac
```bash
chmod +x deploy-helm.sh
./deploy-helm.sh deploy
```

## Prerequisites

1. Helm 3.x
2. kubectl configured for your cluster
3. Slack Bot Token
4. Slack Signing Secret

## Documentation

See HELM_DEPLOYMENT_GUIDE.md for complete instructions.

Generated: $(Get-Date)
"@
    
    Set-Content -Path "$bundleDir\README.md" -Value $readmeContent -Force
    Write-Success "Bundle created: $bundleDir"
    
    # Create compressed archive
    Write-Info "Creating compressed archive..."
    $archiveName = "rag-app-slack-helm-$timestamp.zip"
    Compress-Archive -Path $bundleDir -DestinationPath "$OutputDir\$archiveName" -Force
    
    Write-Header "Helm Deployment Package Created"
    
    Write-Host "Package Details:" -ForegroundColor Cyan
    $archiveItem = Get-Item "$OutputDir\$archiveName"
    Write-Host "  Location: $($archiveItem.FullName)"
    Write-Host "  Size: $([Math]::Round($archiveItem.Length / 1MB, 2)) MB"
    
    Write-Host "`nChart Package:" -ForegroundColor Cyan
    Write-Host "  File: $($package.Name)"
    Write-Host "  Size: $([Math]::Round($package.Length / 1MB, 2)) MB"
    
    Write-Host "`nTo Use:" -ForegroundColor Cyan
    Write-Host "  1. Extract: $archiveName"
    Write-Host "  2. Read: README.md"
    Write-Host "  3. Deploy: .\Deploy-Helm.ps1 -Action deploy"
    
    Write-Success "Ready for distribution!"
    
    # Cleanup
    Remove-Item -Path $bundleDir -Recurse -Force
}
else {
    Write-Warning "No Helm packages found"
    exit 1
}
