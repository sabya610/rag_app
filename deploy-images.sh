#!/bin/bash
# Multi-image deployment and registry push script
# Usage: ./deploy-images.sh [action] [registry]

set -e

REGISTRY="${2:-your-registry.azurecr.io}"
ACTION="${1:-help}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== RAG App Multi-Image Deployment Script ===${NC}\n"

# Functions
show_help() {
    cat << EOF
Usage: $0 [action] [registry]

Actions:
  build-all       Build both SFDC+PDF and Slack-integrated images
  build-sfdc      Build SFDC+PDF image only
  build-slack     Build Slack-integrated image only
  push-all        Push both images to registry
  push-sfdc       Push SFDC+PDF image to registry
  push-slack      Push Slack-integrated image to registry
  test-sfdc       Test SFDC+PDF stack locally
  test-slack      Test Slack-integrated stack locally
  deploy-sfdc     Deploy SFDC+PDF to host
  deploy-slack    Deploy Slack-integrated to host
  rollback        Stop all and revert to SFDC+PDF
  status          Show current deployment status
  clean           Remove all containers and volumes
  help            Show this help message

Registry (default: your-registry.azurecr.io):
  Specify Azure ACR, Docker Hub, or other registry

Examples:
  $0 build-all your-registry.azurecr.io
  $0 push-all your-registry.azurecr.io
  $0 test-sfdc
  $0 deploy-sfdc

EOF
}

build_sfdc() {
    echo -e "${BLUE}[BUILD] SFDC + PDF image...${NC}"
    git checkout main >/dev/null 2>&1 || git checkout main
    docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .
    docker tag rag-app:1.0-sfdc-pdf rag-app:latest-stable
    echo -e "${GREEN}✓ SFDC + PDF image built${NC}\n"
}

build_slack() {
    echo -e "${BLUE}[BUILD] Slack-integrated image...${NC}"
    git checkout feature/slack-integration >/dev/null 2>&1 || git checkout feature/slack-integration
    docker build -f Dockerfile.slack-integrated -t rag-app:1.0-slack-integrated .
    docker tag rag-app:1.0-slack-integrated rag-app:latest-development
    echo -e "${GREEN}✓ Slack-integrated image built${NC}\n"
}

push_sfdc() {
    echo -e "${BLUE}[PUSH] SFDC + PDF image to registry...${NC}"
    docker tag rag-app:1.0-sfdc-pdf $REGISTRY/rag-app:1.0-sfdc-pdf
    docker tag rag-app:1.0-sfdc-pdf $REGISTRY/rag-app:latest-stable
    docker push $REGISTRY/rag-app:1.0-sfdc-pdf
    docker push $REGISTRY/rag-app:latest-stable
    echo -e "${GREEN}✓ SFDC + PDF image pushed${NC}\n"
}

push_slack() {
    echo -e "${BLUE}[PUSH] Slack-integrated image to registry...${NC}"
    docker tag rag-app:1.0-slack-integrated $REGISTRY/rag-app:1.0-slack-integrated
    docker tag rag-app:1.0-slack-integrated $REGISTRY/rag-app:latest-development
    docker push $REGISTRY/rag-app:1.0-slack-integrated
    docker push $REGISTRY/rag-app:latest-development
    echo -e "${GREEN}✓ Slack-integrated image pushed${NC}\n"
}

test_sfdc() {
    echo -e "${BLUE}[TEST] SFDC + PDF stack...${NC}"
    git checkout main >/dev/null 2>&1
    docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
    echo -e "${YELLOW}Waiting for startup...${NC}"
    sleep 10
    
    if curl -s http://localhost:5000/api/rag/stats | grep -q "total_chunks"; then
        echo -e "${GREEN}✓ SFDC + PDF stack is healthy${NC}"
        echo "Access: http://localhost:5000"
    else
        echo -e "${RED}✗ SFDC + PDF stack health check failed${NC}"
        docker logs rag-app-sfdc-pdf | tail -20
    fi
    echo
}

test_slack() {
    echo -e "${BLUE}[TEST] Slack-integrated stack...${NC}"
    git checkout feature/slack-integration >/dev/null 2>&1
    docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d
    
    echo -e "${YELLOW}Waiting for startup...${NC}"
    sleep 10
    
    echo -e "${YELLOW}Initializing Slack database...${NC}"
    docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py
    
    echo -e "${YELLOW}Waiting for DB initialization...${NC}"
    sleep 5
    
    if curl -s http://localhost:5001/api/slack/stats | grep -q "total_messages"; then
        echo -e "${GREEN}✓ Slack-integrated stack is healthy${NC}"
        echo "Access: http://localhost:5001"
    else
        echo -e "${RED}✗ Slack-integrated stack health check failed${NC}"
        docker logs rag-app-slack-integrated | tail -20
    fi
    echo
}

deploy_sfdc() {
    echo -e "${BLUE}[DEPLOY] SFDC + PDF to production...${NC}"
    git checkout main >/dev/null 2>&1
    docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
    
    echo -e "${YELLOW}Waiting for startup...${NC}"
    sleep 10
    
    STATUS=$(curl -s http://localhost:5000/api/rag/stats)
    if echo "$STATUS" | grep -q "total_chunks"; then
        echo -e "${GREEN}✓ SFDC + PDF deployed and running${NC}"
        echo "Endpoint: http://localhost:5000"
        echo "Status: $STATUS"
    else
        echo -e "${RED}✗ Deployment failed${NC}"
        docker logs rag-app-sfdc-pdf
        exit 1
    fi
    echo
}

deploy_slack() {
    echo -e "${BLUE}[DEPLOY] Slack-integrated to testing...${NC}"
    git checkout feature/slack-integration >/dev/null 2>&1
    docker-compose -f docker-compose.slack-integrated.yml --env-file .env.test-slack up -d
    
    echo -e "${YELLOW}Waiting for startup...${NC}"
    sleep 10
    
    echo -e "${YELLOW}Initializing Slack database...${NC}"
    docker-compose -f docker-compose.slack-integrated.yml exec rag-app python init_slack_db.py
    
    echo -e "${YELLOW}Waiting for DB initialization...${NC}"
    sleep 5
    
    STATUS=$(curl -s http://localhost:5001/api/slack/stats)
    if echo "$STATUS" | grep -q "total_messages"; then
        echo -e "${GREEN}✓ Slack-integrated deployed and running${NC}"
        echo "Endpoint: http://localhost:5001"
        echo "Status: $STATUS"
    else
        echo -e "${RED}✗ Deployment failed${NC}"
        docker logs rag-app-slack-integrated
        exit 1
    fi
    echo
}

rollback() {
    echo -e "${BLUE}[ROLLBACK] To SFDC + PDF...${NC}"
    echo -e "${YELLOW}Stopping all containers...${NC}"
    docker-compose -f docker-compose.slack-integrated.yml down 2>/dev/null || true
    
    echo -e "${YELLOW}Switching to main branch...${NC}"
    git checkout main >/dev/null 2>&1
    
    echo -e "${YELLOW}Deploying SFDC + PDF...${NC}"
    docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
    
    sleep 10
    
    if curl -s http://localhost:5000/api/rag/stats | grep -q "total_chunks"; then
        echo -e "${GREEN}✓ Rollback successful${NC}"
        echo "Running: SFDC + PDF on http://localhost:5000"
    else
        echo -e "${RED}✗ Rollback failed${NC}"
    fi
    echo
}

show_status() {
    echo -e "${BLUE}[STATUS] Current deployment...${NC}\n"
    
    echo "Docker containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep rag-app || echo "  None running"
    echo
    
    echo "Docker images:"
    docker images | grep rag-app || echo "  None built"
    echo
    
    echo "Health checks:"
    
    if docker ps | grep -q rag-app-sfdc-pdf; then
        if curl -s http://localhost:5000/api/rag/stats >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} SFDC + PDF: http://localhost:5000"
        else
            echo -e "  ${RED}✗${NC} SFDC + PDF: Not responding"
        fi
    else
        echo -e "  ${YELLOW}○${NC} SFDC + PDF: Not running"
    fi
    
    if docker ps | grep -q rag-app-slack-integrated; then
        if curl -s http://localhost:5001/api/slack/stats >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Slack: http://localhost:5001"
        else
            echo -e "  ${RED}✗${NC} Slack: Not responding"
        fi
    else
        echo -e "  ${YELLOW}○${NC} Slack: Not running"
    fi
    echo
}

clean() {
    echo -e "${BLUE}[CLEAN] Removing all containers and volumes...${NC}"
    
    read -p "This will DELETE all data. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f docker-compose.sfdc-pdf.yml down -v
        docker-compose -f docker-compose.slack-integrated.yml down -v
        docker system prune -f
        echo -e "${GREEN}✓ Cleanup complete${NC}\n"
    else
        echo -e "${YELLOW}Cleanup cancelled${NC}\n"
    fi
}

# Main
case "$ACTION" in
    build-all)
        build_sfdc
        build_slack
        ;;
    build-sfdc)
        build_sfdc
        ;;
    build-slack)
        build_slack
        ;;
    push-all)
        push_sfdc
        push_slack
        ;;
    push-sfdc)
        push_sfdc
        ;;
    push-slack)
        push_slack
        ;;
    test-sfdc)
        test_sfdc
        ;;
    test-slack)
        test_slack
        ;;
    deploy-sfdc)
        deploy_sfdc
        ;;
    deploy-slack)
        deploy_slack
        ;;
    rollback)
        rollback
        ;;
    status)
        show_status
        ;;
    clean)
        clean
        ;;
    *)
        show_help
        ;;
esac
