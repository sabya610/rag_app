# PowerShell Helm Deployment Script for RAG App Slack Integration
# Deploy to FTC AIE 1.1.1 Kubernetes Cluster

param(
    [string]$Action = "deploy",
    [string]$ReleaseName = "rag-app-slack",
    [string]$Namespace = "default",
    [string]$ChartPath = "./helm/rag-app-slack",
    [string]$ValuesFile = "./helm/values-ftc-aie.yaml"
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

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Cyan
}

# Verify Prerequisites
function Test-Prerequisites {
    Write-Header "Verifying Prerequisites"
    
    # Check Helm
    if (-not (Get-Command helm -ErrorAction SilentlyContinue)) {
        Write-Error "Helm not found. Please install Helm 3.x"
        exit 1
    }
    Write-Success "Helm is installed"
    
    # Check kubectl
    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        Write-Error "kubectl not found. Please install kubectl"
        exit 1
    }
    Write-Success "kubectl is installed"
    
    # Check cluster connectivity
    try {
        kubectl cluster-info | Out-Null
        Write-Success "Connected to Kubernetes cluster"
    }
    catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    # Check Helm chart exists
    if (-not (Test-Path "$ChartPath/Chart.yaml")) {
        Write-Error "Helm chart not found at $ChartPath"
        exit 1
    }
    Write-Success "Helm chart found"
}

# Validate credentials
function Get-Credentials {
    Write-Header "Validating Credentials"
    
    $slackBotToken = Read-Host "Enter Slack Bot Token (xoxb-...)"
    if ($slackBotToken -notmatch "^xoxb-") {
        Write-Error "Invalid Slack Bot Token format"
        exit 1
    }
    Write-Success "Slack Bot Token validated"
    
    $slackSecret = Read-Host "Enter Slack Signing Secret"
    if ($slackSecret.Length -lt 32) {
        Write-Error "Invalid Slack Signing Secret"
        exit 1
    }
    Write-Success "Slack Signing Secret validated"
    
    $sfdcUsername = Read-Host "Enter Salesforce Username (or press Enter to skip)"
    $sfdcPassword = Read-Host "Enter Salesforce Password (or press Enter to skip)" -AsSecureString
    $sfdcToken = Read-Host "Enter Salesforce Security Token (or press Enter to skip)"
    
    return @{
        slackBotToken = $slackBotToken
        slackSecret = $slackSecret
        sfdcUsername = $sfdcUsername
        sfdcPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToCoTaskMemUnicode($sfdcPassword))
        sfdcToken = $sfdcToken
    }
}

# Add Helm repos
function Add-HelmRepositories {
    Write-Header "Adding Helm Repositories"
    
    $repos = helm repo list 2>$null
    
    if ($repos -notmatch "bitnami") {
        Write-Info "Adding Bitnami Helm repository..."
        helm repo add bitnami https://charts.bitnami.com/bitnami
    }
    
    helm repo update
    Write-Success "Helm repositories updated"
}

# Lint chart
function Invoke-HelmLint {
    Write-Header "Linting Helm Chart"
    
    helm lint $ChartPath
    Write-Success "Chart validation passed"
}

# Create namespace
function New-Namespace {
    Write-Info "Creating namespace $Namespace if needed..."
    
    $ns = kubectl get namespace $Namespace 2>$null
    if ($null -eq $ns) {
        kubectl create namespace $Namespace | Out-Null
        Write-Success "Namespace created"
    }
    else {
        Write-Success "Namespace already exists"
    }
}

# Deploy with Helm
function Deploy-WithHelm {
    param($Credentials)
    
    Write-Header "Deploying with Helm"
    
    Write-Info "Building Helm command..."
    
    $helmArgs = @(
        "upgrade", "--install", $ReleaseName, $ChartPath,
        "-f", $ValuesFile,
        "--namespace", $Namespace,
        "--set", "slack.botToken=$($Credentials.slackBotToken)",
        "--set", "slack.signingSecret=$($Credentials.slackSecret)"
    )
    
    if ($Credentials.sfdcUsername) {
        $helmArgs += "--set"
        $helmArgs += "salesforce.username=$($Credentials.sfdcUsername)"
    }
    
    if ($Credentials.sfdcPassword) {
        $helmArgs += "--set"
        $helmArgs += "salesforce.password=$($Credentials.sfdcPassword)"
    }
    
    if ($Credentials.sfdcToken) {
        $helmArgs += "--set"
        $helmArgs += "salesforce.securityToken=$($Credentials.sfdcToken)"
    }
    
    $helmArgs += "--wait"
    $helmArgs += "--timeout"
    $helmArgs += "5m"
    
    Write-Host "Executing helm with the provided credentials..." -ForegroundColor Yellow
    
    $response = Read-Host "Proceed with deployment? (yes/no)"
    if ($response -ne "yes") {
        Write-Warning "Deployment cancelled"
        return $false
    }
    
    & helm @helmArgs
    Write-Success "Helm deployment completed"
    return $true
}

# Verify deployment
function Test-Deployment {
    Write-Header "Verifying Deployment"
    
    Write-Info "Waiting for pods to be ready..."
    Start-Sleep -Seconds 5
    
    try {
        $deployment = kubectl get deployment $ReleaseName -n $Namespace
        Write-Success "Deployment created"
        
        # Get pod status
        $ready = kubectl get deployment $ReleaseName -n $Namespace -o jsonpath='{.status.readyReplicas}' 2>$null
        $desired = kubectl get deployment $ReleaseName -n $Namespace -o jsonpath='{.status.desiredReplicas}' 2>$null
        
        Write-Host "Pods: $ready/$desired ready"
        
        Write-Host "`nPod Status:" -ForegroundColor Yellow
        kubectl get pods -n $Namespace -l "app.kubernetes.io/instance=$ReleaseName"
    }
    catch {
        Write-Error "Deployment verification failed"
        return $false
    }
}

# Show deployment info
function Show-DeploymentInfo {
    Write-Header "Deployment Information"
    
    Write-Host "Release: $ReleaseName" -ForegroundColor Yellow
    Write-Host "Namespace: $Namespace" -ForegroundColor Yellow
    Write-Host "Chart: $ChartPath" -ForegroundColor Yellow
    
    try {
        $serviceIP = kubectl get svc $ReleaseName -n $Namespace -o jsonpath='{.spec.clusterIP}' 2>$null
        $servicePort = kubectl get svc $ReleaseName -n $Namespace -o jsonpath='{.spec.ports[0].port}' 2>$null
        
        Write-Host "`nService Access:" -ForegroundColor Yellow
        Write-Host "  Cluster IP: $serviceIP"
        Write-Host "  Port: $servicePort"
        Write-Host "  URL: http://$($serviceIP):$servicePort"
        
        Write-Host "`nAPI Endpoints:" -ForegroundColor Yellow
        Write-Host "  Stats: GET /api/slack/stats"
        Write-Host "  Channels: GET /api/slack/channels"
        Write-Host "  Import: POST /api/slack/import"
        Write-Host "  Search: POST /api/slack/search"
        Write-Host "  Threads: GET /api/slack/threads/{id}"
        
        Write-Host "`nUseful Commands:" -ForegroundColor Yellow
        Write-Host "  View logs: kubectl logs -n $Namespace -l app.kubernetes.io/instance=$ReleaseName -f"
        Write-Host "  Get status: helm status $ReleaseName -n $Namespace"
        Write-Host "  Port forward: kubectl port-forward -n $Namespace svc/$ReleaseName 5001:5001"
        Write-Host "  Uninstall: helm uninstall $ReleaseName -n $Namespace"
    }
    catch {
        Write-Warning "Could not retrieve deployment details"
    }
}

# Package Helm chart
function Invoke-HelmPackage {
    Write-Header "Packaging Helm Chart"
    
    helm package $ChartPath --destination ./helm
    
    $package = Get-ChildItem -Path "./helm/rag-app-slack-*.tgz" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($package) {
        Write-Success "Chart packaged: $($package.Name)"
        Write-Host "Download: $($package.FullName)"
    }
}

# Main function
function Main {
    Write-Header "RAG App Slack Integration - Helm Deployment"
    
    switch ($Action) {
        "deploy" {
            Test-Prerequisites
            Add-HelmRepositories
            Invoke-HelmLint
            New-Namespace
            $creds = Get-Credentials
            if (Deploy-WithHelm -Credentials $creds) {
                Test-Deployment
                Show-DeploymentInfo
                Write-Success "Deployment completed successfully!"
            }
        }
        
        "uninstall" {
            Write-Header "Uninstalling Helm Release"
            $response = Read-Host "Uninstall $ReleaseName from $Namespace? (yes/no)"
            if ($response -eq "yes") {
                helm uninstall $ReleaseName -n $Namespace
                Write-Success "Release uninstalled"
            }
            else {
                Write-Warning "Uninstall cancelled"
            }
        }
        
        "package" {
            Invoke-HelmPackage
        }
        
        "status" {
            helm status $ReleaseName -n $Namespace
        }
        
        default {
            Write-Host "Usage: .\Deploy-Helm.ps1 -Action {deploy|uninstall|package|status}" -ForegroundColor Yellow
            Write-Host "Example: .\Deploy-Helm.ps1 -Action deploy -Namespace default" -ForegroundColor Yellow
        }
    }
}

Main
