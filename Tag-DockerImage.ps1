#!/usr/bin/env pwsh

# Tag the built image with multiple tags

$imageName = "rag-app:latest"
$registry = "sabya610"
$appName = "rag-app"

Write-Host "`n===================================================================="
Write-Host "TAGGING DOCKER IMAGE SEPARATELY"
Write-Host "===================================================================="

Write-Host "`nChecking if image exists..."
$imageExists = docker images | Select-String $imageName
if (-not $imageExists) {
    Write-Host "  ERROR: Image $imageName not found!"
    exit 1
}
Write-Host "  OK: Image $imageName found"

# Get image ID
Write-Host "`nGetting image ID..."
$imageId = docker images | Select-String $imageName | ForEach-Object { ($_ -split '\s+')[2] }
Write-Host "  Image ID: $imageId"

# Create tags
$dateTag = Get-Date -Format "yyyyMMdd"
$datetimeTag = Get-Date -Format "yyyyMMdd_HHmmss"
$tags = @(
    "${registry}/${appName}:latest",
    "${registry}/${appName}:1.0.0",
    "${registry}/${appName}:1.0.1",
    "${registry}/${appName}:${dateTag}",
    "${registry}/${appName}:${datetimeTag}"
)

Write-Host "`nCreating tags..."
foreach ($tag in $tags) {
    Write-Host "  - Tagging as: $tag"
    docker tag $imageName $tag
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    ERROR: Failed"
    } else {
        Write-Host "    OK: Success"
    }
}

Write-Host "`n===================================================================="
Write-Host "ALL TAGS CREATED"
Write-Host "===================================================================="

Write-Host "`nImages with all tags:"
docker images | Select-String $appName | Format-Table

Write-Host "`nNext steps:"
Write-Host "  1. Login to Docker registry:"
Write-Host "     docker login -u sabya610"
Write-Host ""
Write-Host "  2. Push images:"
foreach ($tag in $tags) {
    Write-Host "     docker push $tag"
}
Write-Host ""
Write-Host "  Or push all at once:"
Write-Host "     docker push -a"

Write-Host "`n===================================================================="
