#!/bin/bash

###############################################################################
# Helm Deployment Script for RAG App Slack Integration
# Deploys to FTC AIE 1.1.1 Kubernetes Cluster
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
HELM_RELEASE_NAME="rag-app-slack"
HELM_CHART_PATH="./helm/rag-app-slack"
HELM_VALUES_FILE="./helm/values-ftc-aie.yaml"
NAMESPACE="default"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"

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

# Verify Prerequisites
verify_prerequisites() {
    print_header "Verifying Prerequisites"
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        print_error "Helm not found. Please install Helm 3.x"
        exit 1
    fi
    print_success "Helm is installed"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl"
        exit 1
    fi
    print_success "kubectl is installed"
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_success "Connected to Kubernetes cluster"
    
    # Check Helm chart exists
    if [ ! -f "$HELM_CHART_PATH/Chart.yaml" ]; then
        print_error "Helm chart not found at $HELM_CHART_PATH"
        exit 1
    fi
    print_success "Helm chart found"
}

# Validate credentials
validate_credentials() {
    print_header "Validating Credentials"
    
    read -p "Enter Slack Bot Token (xoxb-...): " SLACK_BOT_TOKEN
    if [[ ! $SLACK_BOT_TOKEN =~ ^xoxb- ]]; then
        print_error "Invalid Slack Bot Token format"
        exit 1
    fi
    print_success "Slack Bot Token validated"
    
    read -p "Enter Slack Signing Secret: " SLACK_SIGNING_SECRET
    if [ -z "$SLACK_SIGNING_SECRET" ] || [ ${#SLACK_SIGNING_SECRET} -lt 32 ]; then
        print_error "Invalid Slack Signing Secret"
        exit 1
    fi
    print_success "Slack Signing Secret validated"
    
    read -p "Enter Salesforce Username (or skip): " SFDC_USERNAME
    read -sp "Enter Salesforce Password (or skip): " SFDC_PASSWORD
    echo
    read -p "Enter Salesforce Security Token (or skip): " SFDC_SECURITY_TOKEN
}

# Add Helm repositories
add_helm_repos() {
    print_header "Adding Helm Repositories"
    
    # Add Bitnami repo for PostgreSQL
    if ! helm repo list | grep -q "bitnami"; then
        print_info "Adding Bitnami Helm repository..."
        helm repo add bitnami https://charts.bitnami.com/bitnami
    fi
    
    helm repo update
    print_success "Helm repositories updated"
}

# Lint Helm chart
lint_chart() {
    print_header "Linting Helm Chart"
    
    helm lint "$HELM_CHART_PATH"
    print_success "Chart validation passed"
}

# Create namespace if needed
create_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_info "Creating namespace $NAMESPACE..."
        kubectl create namespace "$NAMESPACE"
        print_success "Namespace created"
    else
        print_success "Namespace already exists"
    fi
}

# Deploy using Helm
deploy_with_helm() {
    print_header "Deploying with Helm"
    
    print_info "Building Helm command..."
    
    HELM_CMD="helm upgrade --install $HELM_RELEASE_NAME $HELM_CHART_PATH \
        -f $HELM_VALUES_FILE \
        --namespace $NAMESPACE \
        --set slack.botToken=$SLACK_BOT_TOKEN \
        --set slack.signingSecret=$SLACK_SIGNING_SECRET"
    
    if [ -n "$SFDC_USERNAME" ]; then
        HELM_CMD="$HELM_CMD --set salesforce.username=$SFDC_USERNAME"
    fi
    
    if [ -n "$SFDC_PASSWORD" ]; then
        HELM_CMD="$HELM_CMD --set salesforce.password=$SFDC_PASSWORD"
    fi
    
    if [ -n "$SFDC_SECURITY_TOKEN" ]; then
        HELM_CMD="$HELM_CMD --set salesforce.securityToken=$SFDC_SECURITY_TOKEN"
    fi
    
    if [ -n "$DOCKER_USERNAME" ]; then
        HELM_CMD="$HELM_CMD --set ragApp.image.repository=$DOCKER_REGISTRY/$DOCKER_USERNAME/rag-app"
    fi
    
    HELM_CMD="$HELM_CMD --wait --timeout 5m"
    
    echo -e "${YELLOW}Executing:${NC}"
    echo $HELM_CMD
    echo
    
    read -p "Proceed with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        return 1
    fi
    
    eval $HELM_CMD
    
    print_success "Helm deployment completed"
}

# Verify deployment
verify_deployment() {
    print_header "Verifying Deployment"
    
    print_info "Waiting for pods to be ready..."
    sleep 5
    
    # Check deployment status
    if kubectl get deployment "$HELM_RELEASE_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_success "Deployment created"
        
        # Get pod status
        READY=$(kubectl get deployment "$HELM_RELEASE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        DESIRED=$(kubectl get deployment "$HELM_RELEASE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.desiredReplicas}')
        
        echo "Pods: $READY/$DESIRED ready"
        
        # Get pods
        echo -e "\n${YELLOW}Pod Status:${NC}"
        kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$HELM_RELEASE_NAME"
    else
        print_error "Deployment not found"
        return 1
    fi
}

# Show deployment info
show_deployment_info() {
    print_header "Deployment Information"
    
    echo -e "${YELLOW}Release:${NC} $HELM_RELEASE_NAME"
    echo -e "${YELLOW}Namespace:${NC} $NAMESPACE"
    echo -e "${YELLOW}Chart:${NC} $HELM_CHART_PATH"
    
    # Get service info
    SERVICE_IP=$(kubectl get svc "$HELM_RELEASE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "pending")
    SERVICE_PORT=$(kubectl get svc "$HELM_RELEASE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "5001")
    
    echo -e "\n${YELLOW}Service Access:${NC}"
    echo "  Cluster IP: $SERVICE_IP"
    echo "  Port: $SERVICE_PORT"
    echo "  URL: http://$SERVICE_IP:$SERVICE_PORT"
    
    echo -e "\n${YELLOW}API Endpoints:${NC}"
    echo "  Stats: GET /api/slack/stats"
    echo "  Channels: GET /api/slack/channels"
    echo "  Import: POST /api/slack/import"
    echo "  Search: POST /api/slack/search"
    echo "  Threads: GET /api/slack/threads/{id}"
    
    echo -e "\n${YELLOW}Useful Commands:${NC}"
    echo "  View logs: kubectl logs -n $NAMESPACE -l app.kubernetes.io/instance=$HELM_RELEASE_NAME -f"
    echo "  Get status: helm status $HELM_RELEASE_NAME -n $NAMESPACE"
    echo "  Port forward: kubectl port-forward -n $NAMESPACE svc/$HELM_RELEASE_NAME 5001:5001"
    echo "  Uninstall: helm uninstall $HELM_RELEASE_NAME -n $NAMESPACE"
}

# Package Helm chart
package_chart() {
    print_header "Packaging Helm Chart"
    
    helm package "$HELM_CHART_PATH" --destination ./helm
    
    PACKAGE_FILE=$(ls -t ./helm/rag-app-slack-*.tgz | head -1)
    print_success "Chart packaged: $PACKAGE_FILE"
    echo "Download: $PACKAGE_FILE"
}

# Main execution
main() {
    print_header "RAG App Slack Integration - Helm Deployment"
    
    # Parse arguments
    ACTION=${1:-deploy}
    
    case $ACTION in
        deploy)
            verify_prerequisites
            add_helm_repos
            lint_chart
            create_namespace
            validate_credentials
            deploy_with_helm
            verify_deployment
            show_deployment_info
            print_success "Deployment completed successfully!"
            ;;
        uninstall)
            print_header "Uninstalling Helm Release"
            read -p "Uninstall $HELM_RELEASE_NAME from $NAMESPACE? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                helm uninstall "$HELM_RELEASE_NAME" -n "$NAMESPACE"
                print_success "Release uninstalled"
            else
                print_warning "Uninstall cancelled"
            fi
            ;;
        package)
            package_chart
            ;;
        status)
            helm status "$HELM_RELEASE_NAME" -n "$NAMESPACE"
            ;;
        *)
            echo "Usage: $0 {deploy|uninstall|package|status}"
            exit 1
            ;;
    esac
}

main "$@"
