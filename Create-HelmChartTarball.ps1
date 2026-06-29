#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

$projectRoot = "c:\Users\malliks\rag_app"
$helmChartDir = Join-Path $projectRoot "helm\rag-app-slack"
$releaseDir = Join-Path $projectRoot "releases"
$tempDir = Join-Path $env:TEMP "helm-chart-build-$(Get-Random)"

Write-Host "`n===================================================================="
Write-Host "CREATING HELM CHART TAR.GZ FOR FRAMEWORK IMPORT"
Write-Host "===================================================================="

New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
Write-Host "`nStep 1: Setting up build directory..."

Copy-Item -Path $helmChartDir -Destination "$tempDir\rag-app-slack" -Recurse
Write-Host "  - Copied Helm chart"

$chartYamlPath = "$tempDir\rag-app-slack\Chart.yaml"
if (Test-Path $chartYamlPath) {
    Write-Host "  - Chart.yaml verified"
} else {
    throw "Chart.yaml not found"
}

Write-Host "`nStep 2: Creating tar.gz archive..."
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tarballName = "rag-app-slack-$timestamp.tar.gz"
$tarballPath = Join-Path $releaseDir $tarballName

Push-Location $tempDir
tar -czf $tarballPath "rag-app-slack"
Pop-Location

Write-Host "  - Archive created"

Write-Host "`nStep 3: Verifying contents..."
$firstFive = tar -tzf $tarballPath | Select-Object -First 5
$firstFive | ForEach-Object { Write-Host "    $_" }

$fileSize = [Math]::Round((Get-Item $tarballPath).Length / 1KB, 2)

Write-Host "`n===================================================================="
Write-Host "SUCCESS!"
Write-Host "===================================================================="
Write-Host "File:  $tarballName"
Write-Host "Size:  $fileSize KB"
Write-Host "Path:  $tarballPath"
Write-Host ""
Write-Host "Chart: rag-app-slack/Chart.yaml"
Write-Host "Ready for CloudViz framework import"
Write-Host "===================================================================="

Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
