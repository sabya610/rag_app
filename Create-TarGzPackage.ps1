param(
    [ValidateSet('chart', 'full')]
    [string]$PackageType = 'chart'
)

$releaseDir = "./releases"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "HELM PACKAGE CREATOR - TAR.GZ FORMAT" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "`n"

# Ensure releases directory exists
if (!(Test-Path $releaseDir)) {
    New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
}

if ($PackageType -eq 'chart') {
    Write-Host "Creating Helm Chart package (tar.gz)..." -ForegroundColor Yellow
    
    $packageName = "rag-app-slack-$timestamp.tar.gz"
    $tarPath = Join-Path $releaseDir $packageName
    
    # Create tar.gz of just the chart directory
    Write-Host "  - Compressing helm/rag-app-slack..." -ForegroundColor Cyan
    
    # Use tar command with full path
    $chartDir = Resolve-Path "./helm/rag-app-slack"
    $fullTarPath = (Resolve-Path $releaseDir).Path + "\" + $packageName
    
    Push-Location "./helm"
    tar -czf $fullTarPath rag-app-slack
    Pop-Location
    
    Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "HELM CHART PACKAGE CREATED" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
    
    $file = Get-Item $fullTarPath
    Write-Host "`nChart Package Details:" -ForegroundColor Green
    Write-Host "  Name:     $($file.Name)" -ForegroundColor Yellow
    Write-Host "  Size:     $([math]::Round($file.Length / 1KB, 2)) KB" -ForegroundColor Yellow
    Write-Host "  Location: $($file.FullName)" -ForegroundColor Yellow
    Write-Host "  Format:   tar.gz (for Kubernetes import)" -ForegroundColor Yellow
    
    Write-Host "`nUsage in Kubernetes:" -ForegroundColor Green
    Write-Host "  1. Navigate to: Framework > Import Framework" -ForegroundColor Cyan
    Write-Host "  2. Select: 'Upload New Chart'" -ForegroundColor Cyan
    Write-Host "  3. Upload file: $($file.Name)" -ForegroundColor Cyan
    Write-Host "  4. Enter Namespace and Release Name" -ForegroundColor Cyan
    Write-Host "  5. Click 'Framework Values' to configure" -ForegroundColor Cyan
    
} else {
    Write-Host "Creating Full Deployment package (tar.gz)..." -ForegroundColor Yellow
    
    $tempDir = "./temp_package"
    $packageName = "rag-app-slack-deployment-$timestamp"
    $fullPackageName = $packageName + ".tar.gz"
    $tarPath = Join-Path $releaseDir $fullPackageName
    
    # Create temporary structure
    if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
    New-Item -ItemType Directory -Path "$tempDir/$packageName" -Force | Out-Null
    
    Write-Host "  - Creating deployment structure..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path "$tempDir/$packageName/helm" -Force | Out-Null
    
    # Copy files
    Write-Host "  - Copying Helm chart..." -ForegroundColor Cyan
    Copy-Item -Path "./helm/rag-app-slack" -Destination "$tempDir/$packageName/helm/" -Recurse -Force
    Copy-Item -Path "./helm/values-ftc-aie.yaml" -Destination "$tempDir/$packageName/helm/" -Force
    Copy-Item -Path "./helm/HELM_DEPLOYMENT_GUIDE.md" -Destination "$tempDir/$packageName/helm/" -Force
    Copy-Item -Path "./helm/QUICK_REFERENCE.md" -Destination "$tempDir/$packageName/helm/" -Force
    Copy-Item -Path "./helm/Deploy-Helm.ps1" -Destination "$tempDir/$packageName/helm/" -Force
    Copy-Item -Path "./helm/deploy-helm.sh" -Destination "$tempDir/$packageName/helm/" -Force
    
    Write-Host "  - Copying documentation..." -ForegroundColor Cyan
    Copy-Item -Path "./DEPLOYMENT_CHECKLIST.md" -Destination "$tempDir/$packageName/" -Force
    Copy-Item -Path "./HELM_PACKAGE_SUMMARY.md" -Destination "$tempDir/$packageName/" -Force
    
    # Create README
    $readmeLines = @(
        "# RAG App Slack Integration - Helm Deployment Package"
        ""
        "Status: Ready for Production Deployment"
        ""
        "## Quick Start"
        ""
        "### Extract Package"
        "tar -xzf $fullPackageName"
        "cd $packageName"
        ""
        "### Deploy to Kubernetes"
        ""
        "#### Windows PowerShell"
        "cd helm"
        ".\Deploy-Helm.ps1 -Action deploy"
        ""
        "#### Linux/Mac/WSL"
        "cd helm"
        "chmod +x deploy-helm.sh"
        "./deploy-helm.sh deploy"
        ""
        "## Documentation"
        "1. START_HERE.md - Quick start"
        "2. HELM_PACKAGE_SUMMARY.md - Overview"
        "3. DEPLOYMENT_CHECKLIST.md - Pre-deployment"
        "4. helm/HELM_DEPLOYMENT_GUIDE.md - Complete guide"
    )
    Set-Content -Path "$tempDir/$packageName/README.md" -Value $readmeLines
    
    # Create START_HERE
    $startLines = @(
        "# START HERE"
        ""
        "## Extract and Deploy"
        ""
        "tar -xzf $fullPackageName"
        "cd $packageName/helm"
        ""
        "# Windows:"
        ".\Deploy-Helm.ps1 -Action deploy"
        ""
        "# Linux/Mac:"
        "./deploy-helm.sh deploy"
        ""
        "## Documentation Order"
        "1. README.md"
        "2. ../HELM_PACKAGE_SUMMARY.md"
        "3. ../DEPLOYMENT_CHECKLIST.md"
        "4. HELM_DEPLOYMENT_GUIDE.md"
    )
    Set-Content -Path "$tempDir/$packageName/START_HERE.md" -Value $startLines
    
    # Create tar.gz
    Write-Host "  - Compressing to tar.gz..." -ForegroundColor Cyan
    $fullTarPath = (Resolve-Path $releaseDir).Path + "\" + $fullPackageName
    Push-Location $tempDir
    tar -czf $fullTarPath $packageName
    Pop-Location
    
    # Cleanup
    Remove-Item $tempDir -Recurse -Force
    
    Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "FULL DEPLOYMENT PACKAGE CREATED" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
    
    $file = Get-Item $fullTarPath
    Write-Host "`nPackage Details:" -ForegroundColor Green
    Write-Host "  Name:     $($file.Name)" -ForegroundColor Yellow
    Write-Host "  Size:     $([math]::Round($file.Length / 1MB, 2)) MB" -ForegroundColor Yellow
    Write-Host "  Location: $($file.FullName)" -ForegroundColor Yellow
    
    Write-Host "`nContents:" -ForegroundColor Green
    Write-Host "  - Complete Helm chart" -ForegroundColor Cyan
    Write-Host "  - Deployment scripts" -ForegroundColor Cyan
    Write-Host "  - FTC configuration" -ForegroundColor Cyan
    Write-Host "  - Full documentation" -ForegroundColor Cyan
}

Write-Host "`nFile Location: releases/$([System.IO.Path]::GetFileName($tarPath))" -ForegroundColor Yellow
Write-Host "`n════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan
