# Slack Integration Deployment Script for FTC AIE 1.1.1 Cluster
# PowerShell Version for Windows
# Deploys SFDC+PDF+Slack integration to distributed hosts via SSH

param(
    [string]$Action = "deploy",
    [string]$DockerUsername = $env:DOCKER_USERNAME,
    [string]$DockerPassword = $env:DOCKER_PASSWORD,
    [string]$SshKeyPath = $env:SSH_KEY_PATH
)

# Configuration
$ClusterName = "FTC-AIE-1.1.1"
$InstallerHost = "10.227.81.151"
$CoordinatorHost = "10.227.81.154"
$MasterHost = "10.227.81.157"
$WorkerHosts = @("10.227.81.160", "10.227.81.161", "10.227.81.162")
$NfsHost = "10.227.81.170"

$SshUser = "root"
$SshPassword = "BDn@nPB42L!"
$DockerRegistry = "docker.io"
$DockerImageTag = "slack-integrated"

$RepoUrl = "https://github.com/sabya610/rag_app.git"
$RepoBranch = "feature/slack-integration"
$DeploymentDir = "/opt/rag_app"

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

# SSH Helper Function
function Invoke-RemoteCommand {
    param(
        [string]$Host,
        [string]$Command,
        [string]$User = $SshUser,
        [string]$Password = $SshPassword
    )
    
    try {
        # Create SSH session
        $session = New-SSHSession -ComputerName $Host -Credential (New-Object PSCredential($User, (ConvertTo-SecureString $Password -AsPlainText -Force))) -ErrorAction Stop
        
        # Invoke command
        $result = Invoke-SSHCommand -SSHSession $session -Command $Command -ErrorAction Stop
        
        # Remove session
        Remove-SSHSession -SSHSession $session | Out-Null
        
        return $result.Output -join "`n"
    }
    catch {
        Write-Error "Failed to execute command on $Host : $_"
        return $null
    }
}

function Test-HostConnectivity {
    Write-Header "Verifying Cluster Connectivity"
    
    $allHosts = @($InstallerHost, $CoordinatorHost, $MasterHost) + $WorkerHosts + @($NfsHost)
    
    foreach ($host in $allHosts) {
        if (Test-Connection -ComputerName $host -Count 1 -Quiet) {
            Write-Success "Host $host is reachable"
        }
        else {
            Write-Error "Cannot reach host $host"
            return $false
        }
    }
    
    return $true
}

function New-DeploymentPlan {
    Write-Header "FTC AIE 1.1.1 Cluster - Slack Integration Deployment Plan"
    
    Write-Info "Cluster Configuration:"
    Write-Info "  Installer: $InstallerHost"
    Write-Info "  Coordinator: $CoordinatorHost"
    Write-Info "  Master: $MasterHost"
    Write-Info "  Workers: $($WorkerHosts -join ', ')"
    Write-Info "  NFS Host: $NfsHost"
    Write-Info "  Repository Branch: $RepoBranch"
    Write-Info "  Deployment Directory: $DeploymentDir"
    
    Write-Host "`nDeployment Steps:" -ForegroundColor Yellow
    Write-Host "1. Verify cluster connectivity"
    Write-Host "2. Deploy prerequisites (Docker, Docker Compose)"
    Write-Host "3. Clone/update Slack integration branch"
    Write-Host "4. Create .env configuration with Slack credentials"
    Write-Host "5. Build Docker images"
    Write-Host "6. Push images to Docker Hub"
    Write-Host "7. Deploy Slack integration stack"
    Write-Host "8. Initialize Slack database"
    Write-Host "9. Test deployment endpoints"
    
    $response = Read-Host "`nProceed with deployment? (yes/no)"
    return $response -eq "yes"
}

function Deploy-Prerequisites {
    param([string]$Host)
    
    Write-Info "Deploying prerequisites to $Host..."
    
    $script = @"
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker root
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

echo "Prerequisites verified"
"@
    
    $output = Invoke-RemoteCommand -Host $Host -Command $script
    if ($output) {
        Write-Success "Prerequisites deployed on $Host"
        return $true
    }
    return $false
}

function Setup-DeploymentDirectory {
    param([string]$Host)
    
    Write-Info "Setting up deployment directory on $Host..."
    
    $script = @"
mkdir -p $DeploymentDir
cd $DeploymentDir

if [ -d ".git" ]; then
    git fetch origin
    git checkout $RepoBranch
    git pull origin $RepoBranch
else
    git clone -b $RepoBranch $RepoUrl .
fi

echo "Repository ready"
"@
    
    $output = Invoke-RemoteCommand -Host $Host -Command $script
    if ($output) {
        Write-Success "Deployment directory set up on $Host"
        return $true
    }
    return $false
}

function Create-EnvFile {
    param([string]$Host)
    
    Write-Header "Creating .env Configuration"
    Write-Warning "Important: You need to provide Slack credentials"
    
    Write-Host "`nRequired Slack Credentials:" -ForegroundColor Yellow
    Write-Host "1. Slack Bot Token (starts with 'xoxb-')"
    Write-Host "   Get from: https://api.slack.com/apps > Your App > Install App"
    Write-Host ""
    Write-Host "2. Slack Signing Secret"
    Write-Host "   Get from: https://api.slack.com/apps > Your App > Basic Information"
    Write-Host ""
    Write-Host "3. Salesforce Credentials (if not already configured)"
    Write-Host ""
    
    $slackToken = Read-Host "Enter Slack Bot Token (xoxb-...)"
    $slackSecret = Read-Host "Enter Slack Signing Secret"
    $sfdcUsername = Read-Host "Enter SFDC Username (or press Enter to skip)"
    $sfdcPassword = Read-Host "Enter SFDC Password (or press Enter to skip)"
    $sfdcSecurityToken = Read-Host "Enter SFDC Security Token (or press Enter to skip)"
    
    $envContent = @"
# Database Configuration
DB_USER=postgres
DB_PASS=postgres123
DB_HOST=postgres-slack-integrated
DB_PORT=5433
DB_NAME=ragdb_slack

# Model Paths
EMBEDDING_MODEL=models/embedding/all-MiniLM-L6-v2
MODEL_PATH=models/llama-2-7b-chat.Q4_K_M.gguf

# Salesforce Configuration
SFDC_USERNAME=$sfdcUsername
SFDC_PASSWORD=$sfdcPassword
SFDC_SECURITY_TOKEN=$sfdcSecurityToken

# Slack Configuration
SLACK_BOT_TOKEN=$slackToken
SLACK_SIGNING_SECRET=$slackSecret

# Import Settings
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30

# Features
FEATURES_SLACK=true

# Server Configuration
FLASK_ENV=production
GUNICORN_WORKERS=4
GUNICORN_PORT=5001
"@
    
    $script = "cat > $DeploymentDir/.env << 'EOF'" + "`n$envContent`nEOF"
    
    Invoke-RemoteCommand -Host $Host -Command $script | Out-Null
    
    Write-Success ".env file created on $Host"
}

function Build-DockerImages {
    param([string]$Host)
    
    Write-Header "Building Docker Images on $Host"
    
    $script = @"
cd $DeploymentDir

echo "Building Slack-integrated image..."
docker build -f Dockerfile.slack-integrated -t rag-app:$DockerImageTag .

if [ ! -z "$DockerUsername" ]; then
    echo "Tagging for Docker Hub..."
    docker tag rag-app:$DockerImageTag $DockerRegistry/$DockerUsername/rag-app:$DockerImageTag
    
    echo "Pushing to Docker Hub..."
    docker push $DockerRegistry/$DockerUsername/rag-app:$DockerImageTag
fi

echo "Image build complete"
"@
    
    $output = Invoke-RemoteCommand -Host $Host -Command $script
    if ($output) {
        Write-Success "Docker images built successfully"
        return $true
    }
    return $false
}

function Deploy-SlackStack {
    param([string]$Host)
    
    Write-Header "Deploying Slack Integration Stack on $Host"
    
    $script = @"
cd $DeploymentDir

echo "Starting Slack integration stack..."
docker-compose -f docker-compose.slack-integrated.yml up -d

echo "Waiting for services to be healthy..."
sleep 15

echo "Initializing Slack database..."
docker-compose -f docker-compose.slack-integrated.yml exec -T rag-app-slack-integrated python init_slack_db.py || true

echo "Checking stack status..."
docker-compose -f docker-compose.slack-integrated.yml ps
"@
    
    $output = Invoke-RemoteCommand -Host $Host -Command $script
    if ($output) {
        Write-Success "Slack integration stack deployed"
        Write-Host $output
        return $true
    }
    return $false
}

function Test-Deployment {
    param([string]$Host)
    
    Write-Header "Testing Deployment on $Host"
    
    Start-Sleep -Seconds 5
    
    try {
        $sfdc = Invoke-WebRequest -Uri "http://${Host}:5000/api/rag/stats" -ErrorAction SilentlyContinue
        if ($sfdc.StatusCode -eq 200) {
            Write-Success "SFDC+PDF endpoint responding"
        }
        else {
            Write-Warning "SFDC+PDF endpoint not responding with 200"
        }
    }
    catch {
        Write-Warning "SFDC+PDF endpoint not accessible (this is OK if in different network)"
    }
    
    try {
        $slack = Invoke-WebRequest -Uri "http://${Host}:5001/api/slack/stats" -ErrorAction SilentlyContinue
        if ($slack.StatusCode -eq 200) {
            Write-Success "Slack endpoint responding"
        }
        else {
            Write-Warning "Slack endpoint not responding with 200"
        }
    }
    catch {
        Write-Warning "Slack endpoint not accessible (this is OK if in different network)"
    }
    
    Write-Success "Deployment tests completed"
}

function Show-DeploymentInfo {
    param([string]$Host)
    
    Write-Header "Deployment Information"
    
    Write-Host "Host: $Host" -ForegroundColor Yellow
    Write-Host "SFDC+PDF: http://$Host`:5000" -ForegroundColor Yellow
    Write-Host "Slack Integration: http://$Host`:5001" -ForegroundColor Yellow
    
    Write-Host "`nAPI Endpoints:" -ForegroundColor Yellow
    Write-Host "  SFDC Stats: GET http://$Host`:5000/api/rag/stats"
    Write-Host "  Slack Stats: GET http://$Host`:5001/api/slack/stats"
    Write-Host "  Slack Import: POST http://$Host`:5001/api/slack/import"
    Write-Host "  Slack Search: POST http://$Host`:5001/api/slack/search"
    Write-Host "  Slack Channels: GET http://$Host`:5001/api/slack/channels"
    Write-Host "  Slack Threads: GET http://$Host`:5001/api/slack/threads/{id}"
    
    Write-Host "`nUseful Commands:" -ForegroundColor Yellow
    Write-Host "  SSH to host: ssh root@$Host"
    Write-Host "  View logs: cd $DeploymentDir && docker-compose -f docker-compose.slack-integrated.yml logs -f rag-app-slack-integrated"
    Write-Host "  Stop stack: docker-compose -f docker-compose.slack-integrated.yml down"
    Write-Host "  Restart stack: docker-compose -f docker-compose.slack-integrated.yml restart"
}

# Main Execution
function Main {
    Write-Header "FTC AIE 1.1.1 Cluster - Slack Integration Deployment"
    
    if (-not (Test-HostConnectivity)) {
        Write-Error "Cluster connectivity check failed"
        exit 1
    }
    
    if (-not (New-DeploymentPlan)) {
        Write-Warning "Deployment cancelled by user"
        exit 0
    }
    
    $deploymentTarget = $InstallerHost
    
    # Deploy prerequisites
    if (-not (Deploy-Prerequisites -Host $deploymentTarget)) {
        Write-Error "Prerequisites deployment failed"
        exit 1
    }
    
    # Setup deployment directory
    if (-not (Setup-DeploymentDirectory -Host $deploymentTarget)) {
        Write-Error "Deployment directory setup failed"
        exit 1
    }
    
    # Create .env file
    Create-EnvFile -Host $deploymentTarget
    
    # Build and push images
    if (-not (Build-DockerImages -Host $deploymentTarget)) {
        Write-Error "Docker build failed"
        exit 1
    }
    
    # Deploy stack
    if (-not (Deploy-SlackStack -Host $deploymentTarget)) {
        Write-Error "Stack deployment failed"
        exit 1
    }
    
    # Test deployment
    Test-Deployment -Host $deploymentTarget
    
    # Show deployment info
    Show-DeploymentInfo -Host $deploymentTarget
    
    Write-Success "Deployment completed successfully!"
}

# Execute main function
Main
