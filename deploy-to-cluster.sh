#!/bin/bash

###############################################################################
# Slack Integration Deployment Script for FTC AIE 1.1.1 Cluster
# Deploys SFDC+PDF+Slack integration to distributed hosts
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="FTC-AIE-1.1.1"
INSTALLER_HOST="10.227.81.151"
COORDINATOR_HOST="10.227.81.154"
MASTER_HOST="10.227.81.157"
WORKER_HOSTS=("10.227.81.160" "10.227.81.161" "10.227.81.162")
NFS_HOST="10.227.81.170"

SSH_USER="root"
SSH_KEY_PATH="${SSH_KEY_PATH:-.}"  # Default to current dir if not set
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
DOCKER_IMAGE_TAG="slack-integrated"

REPO_URL="https://github.com/sabya610/rag_app.git"
REPO_BRANCH="feature/slack-integration"
DEPLOYMENT_DIR="/opt/rag_app"

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

verify_connectivity() {
    print_header "Verifying Cluster Connectivity"
    
    local hosts=("$INSTALLER_HOST" "$COORDINATOR_HOST" "$MASTER_HOST" "${WORKER_HOSTS[@]}" "$NFS_HOST")
    
    for host in "${hosts[@]}"; do
        if ping -c 1 "$host" &> /dev/null; then
            print_success "Host $host is reachable"
        else
            print_error "Cannot reach host $host"
            return 1
        fi
    done
}

verify_docker_hub() {
    print_header "Verifying Docker Hub Configuration"
    
    if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PASSWORD" ]; then
        print_warning "Docker Hub credentials not set in environment variables"
        echo "Please set: export DOCKER_USERNAME=<your_docker_username>"
        echo "            export DOCKER_PASSWORD=<your_docker_password>"
        read -p "Continue without pushing to Docker Hub? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            DOCKER_REGISTRY="local"
            print_warning "Using local Docker images only"
        else
            return 1
        fi
    else
        print_success "Docker Hub credentials configured"
    fi
}

deploy_to_host() {
    local host=$1
    local role=$2
    
    print_header "Deploying to $role ($host)"
    
    ssh -o StrictHostKeyChecking=no "$SSH_USER@$host" bash << 'EOF'
        set -e
        
        # Check Docker
        if ! command -v docker &> /dev/null; then
            echo "Installing Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker root
        fi
        
        # Check Docker Compose
        if ! command -v docker-compose &> /dev/null; then
            echo "Installing Docker Compose..."
            sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        fi
        
        echo "Deployment prerequisites verified on $host"
EOF
    
    print_success "Prerequisites verified on $role ($host)"
}

setup_deployment_directory() {
    local host=$1
    local role=$2
    
    print_info "Setting up deployment directory on $role ($host)..."
    
    ssh -o StrictHostKeyChecking=no "$SSH_USER@$host" bash << EOF
        set -e
        
        if [ ! -d "$DEPLOYMENT_DIR" ]; then
            echo "Creating deployment directory..."
            mkdir -p "$DEPLOYMENT_DIR"
        fi
        
        cd "$DEPLOYMENT_DIR"
        
        # Clone or update repository
        if [ -d ".git" ]; then
            echo "Updating existing repository..."
            git fetch origin
            git checkout $REPO_BRANCH
            git pull origin $REPO_BRANCH
        else
            echo "Cloning repository..."
            git clone -b $REPO_BRANCH "$REPO_URL" .
        fi
        
        echo "Repository ready at $DEPLOYMENT_DIR"
EOF
    
    print_success "Deployment directory set up on $role ($host)"
}

create_env_file() {
    local host=$1
    local env_file=$2
    
    print_info "Creating .env file on $host..."
    
    ssh -o StrictHostKeyChecking=no "$SSH_USER@$host" bash << EOF
        cat > "$DEPLOYMENT_DIR/$env_file" << 'ENVEOF'
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
SFDC_USERNAME=your_sfdc_username
SFDC_PASSWORD=your_sfdc_password
SFDC_SECURITY_TOKEN=your_security_token
SFDC_CLIENT_ID=your_client_id
SFDC_CLIENT_SECRET=your_client_secret

# Slack Configuration
SLACK_BOT_TOKEN=xoxb-YOUR_SLACK_BOT_TOKEN
SLACK_SIGNING_SECRET=YOUR_SIGNING_SECRET

# Import Settings
SLACK_IMPORT_LIMIT=100
SLACK_IMPORT_DAYS=30

# Features
FEATURES_SLACK=true

# Server Configuration
FLASK_ENV=production
GUNICORN_WORKERS=4
GUNICORN_PORT=5001

# Redis Cache (optional, for future caching)
REDIS_URL=redis://redis:6379

ENVEOF
        chmod 600 "$DEPLOYMENT_DIR/$env_file"
        echo ".env file created at $DEPLOYMENT_DIR/$env_file"
EOF
    
    print_warning "⚠ IMPORTANT: Update .env file with actual credentials on host $host"
    print_info "Run: ssh root@$host"
    print_info "Then: nano $DEPLOYMENT_DIR/$env_file"
}

docker_login() {
    if [ "$DOCKER_REGISTRY" != "local" ]; then
        print_header "Docker Hub Login"
        
        local host=$1
        
        ssh -o StrictHostKeyChecking=no "$SSH_USER@$host" bash << EOF
            echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
            echo "Docker Hub login successful"
EOF
        
        print_success "Docker Hub login successful on $host"
    fi
}

build_images() {
    local host=$1
    
    print_header "Building Docker Images on $host"
    
    ssh -o StrictHostKeyChecking=no "$SSH_USER@$host" bash << EOF
        set -e
        cd "$DEPLOYMENT_DIR"
        
        echo "Building Slack-integrated image..."
        docker build -f Dockerfile.slack-integrated -t rag-app:$DOCKER_IMAGE_TAG .
        
        if [ "$DOCKER_REGISTRY" != "local" ]; then
            echo "Tagging for Docker Hub..."
            docker tag rag-app:$DOCKER_IMAGE_TAG "$DOCKER_REGISTRY/$DOCKER_USERNAME/rag-app:$DOCKER_IMAGE_TAG"
            
            echo "Pushing to Docker Hub..."
            docker push "$DOCKER_REGISTRY/$DOCKER_USERNAME/rag-app:$DOCKER_IMAGE_TAG"
        fi
        
        echo "Image build complete"
EOF
    
    print_success "Docker images built successfully"
}

deploy_slack_stack() {
    local host=$1
    
    print_header "Deploying Slack Integration Stack on $host"
    
    ssh -o StrictHostKeyChecking=no "$SSH_USER@$host" bash << EOF
        set -e
        cd "$DEPLOYMENT_DIR"
        
        echo "Starting Slack integration stack..."
        docker-compose -f docker-compose.slack-integrated.yml up -d
        
        echo "Waiting for services to be healthy..."
        sleep 10
        
        echo "Initializing Slack database..."
        docker-compose -f docker-compose.slack-integrated.yml exec -T rag-app-slack-integrated python init_slack_db.py || true
        
        echo "Checking stack status..."
        docker-compose -f docker-compose.slack-integrated.yml ps
EOF
    
    print_success "Slack integration stack deployed"
}

test_deployment() {
    local host=$1
    local port=$2
    
    print_header "Testing Deployment on $host:$port"
    
    ssh -o StrictHostKeyChecking=no "$SSH_USER@$host" bash << EOF
        echo "Testing health endpoints..."
        
        # Test SFDC endpoint
        if curl -s http://localhost:5000/api/rag/stats | grep -q "rag_version"; then
            echo "✓ SFDC+PDF endpoint responding"
        else
            echo "✗ SFDC+PDF endpoint not responding"
        fi
        
        # Test Slack endpoint
        if curl -s http://localhost:5001/api/slack/stats | grep -q "slack_version"; then
            echo "✓ Slack endpoint responding"
        else
            echo "✗ Slack endpoint not responding"
        fi
        
        # List running containers
        echo "Running containers:"
        docker ps --filter "name=rag-app" --format "table {{.Names}}\t{{.Status}}"
EOF
    
    print_success "Deployment tests completed"
}

show_deployment_info() {
    local host=$1
    
    print_header "Deployment Information"
    
    echo -e "${YELLOW}Host: $host${NC}"
    echo -e "${YELLOW}SFDC+PDF: http://$host:5000${NC}"
    echo -e "${YELLOW}Slack Integration: http://$host:5001${NC}"
    echo -e "\n${YELLOW}API Endpoints:${NC}"
    echo "  SFDC: GET http://$host:5000/api/rag/stats"
    echo "  Slack Stats: GET http://$host:5001/api/slack/stats"
    echo "  Slack Import: POST http://$host:5001/api/slack/import"
    echo "  Slack Search: POST http://$host:5001/api/slack/search"
    echo "  Slack Channels: GET http://$host:5001/api/slack/channels"
    echo "  Slack Threads: GET http://$host:5001/api/slack/threads/{id}"
    echo -e "\n${YELLOW}Logs:${NC}"
    echo "  SSH to $host and run:"
    echo "  docker-compose -f $DEPLOYMENT_DIR/docker-compose.slack-integrated.yml logs -f rag-app-slack-integrated"
}

main() {
    print_header "FTC AIE 1.1.1 Cluster - Slack Integration Deployment"
    
    print_info "Cluster Configuration:"
    print_info "  Installer: $INSTALLER_HOST"
    print_info "  Coordinator: $COORDINATOR_HOST"
    print_info "  Master: $MASTER_HOST"
    print_info "  Workers: ${WORKER_HOSTS[*]}"
    print_info "  NFS Host: $NFS_HOST"
    print_info "  Repository Branch: $REPO_BRANCH"
    
    # Deployment will be on Installer host (control node)
    DEPLOYMENT_TARGET=$INSTALLER_HOST
    
    read -p "Proceed with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Deployment cancelled"
        return 1
    fi
    
    # Execute deployment steps
    verify_connectivity || { print_error "Cluster connectivity check failed"; exit 1; }
    verify_docker_hub || { print_error "Docker Hub verification failed"; exit 1; }
    
    deploy_to_host "$DEPLOYMENT_TARGET" "Installer" || { print_error "Prerequisites installation failed"; exit 1; }
    setup_deployment_directory "$DEPLOYMENT_TARGET" "Installer" || { print_error "Deployment directory setup failed"; exit 1; }
    
    create_env_file "$DEPLOYMENT_TARGET" ".env"
    
    read -p "Have you updated the .env file with credentials? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Please update .env file and run deployment again"
        return 1
    fi
    
    docker_login "$DEPLOYMENT_TARGET"
    build_images "$DEPLOYMENT_TARGET" || { print_error "Docker build failed"; exit 1; }
    deploy_slack_stack "$DEPLOYMENT_TARGET" || { print_error "Stack deployment failed"; exit 1; }
    test_deployment "$DEPLOYMENT_TARGET" 5001
    
    show_deployment_info "$DEPLOYMENT_TARGET"
    
    print_success "Deployment completed successfully!"
}

# Execute main function
main "$@"
