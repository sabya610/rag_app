#!/bin/bash
# Production deployment script for SFDC + PDF RAG App
# Usage: ./deploy-images.sh [action]

set -e

ACTION="${1:-help}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== RAG App Deployment Script ===${NC}\n"

show_help() {
    cat << EOF
Usage: $0 [action]

Actions:
  build           Build SFDC+PDF production image
  deploy          Deploy SFDC+PDF stack locally
  status          Show deployment status
  test            Run health checks
  logs            Show application logs
  push [registry] Push image to registry (e.g., your-registry.azurecr.io)
  stop            Stop all containers
  clean           Remove all containers and volumes
  help            Show this help message

Examples:
  $0 build
  $0 deploy
  $0 push your-registry.azurecr.io
  $0 status

EOF
}

build() {
    echo -e "${BLUE}[BUILD] SFDC + PDF production image...${NC}"
    git checkout main >/dev/null 2>&1 || echo "Already on main"
    docker build -f Dockerfile.sfdc-pdf -t rag-app:1.0-sfdc-pdf .
    docker tag rag-app:1.0-sfdc-pdf rag-app:latest
    echo -e "${GREEN}✓ Production image built${NC}\n"
}

deploy() {
    echo -e "${BLUE}[DEPLOY] SFDC + PDF stack...${NC}"
    
    if [ ! -f ".env.demo-sfdc" ]; then
        echo -e "${RED}✗ .env.demo-sfdc not found${NC}"
        echo -e "${YELLOW}Create it with: cp .env.template .env.demo-sfdc${NC}"
        exit 1
    fi
    
    docker-compose -f docker-compose.sfdc-pdf.yml --env-file .env.demo-sfdc up -d
    echo -e "${YELLOW}Waiting for startup...${NC}"
    sleep 10
    
    if curl -s http://localhost:5000/api/rag/stats | grep -q "total_chunks"; then
        echo -e "${GREEN}✓ Stack deployed and running${NC}"
        echo "Access: http://localhost:5000"
        curl http://localhost:5000/api/rag/stats
    else
        echo -e "${RED}✗ Deployment failed${NC}"
        docker logs rag-app-sfdc-pdf | tail -20
        exit 1
    fi
    echo
}

status() {
    echo -e "${BLUE}[STATUS] Current deployment...${NC}\n"
    
    echo "Containers:"
    docker ps --filter "name=rag-app-sfdc-pdf" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "  None running"
    echo
    
    echo "Images:"
    docker images | grep rag-app | head -3 || echo "  None built"
    echo
    
    echo "Health:"
    if docker ps | grep -q rag-app-sfdc-pdf; then
        if curl -s http://localhost:5000/api/rag/stats >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} App is responding on http://localhost:5000"
        else
            echo -e "  ${RED}✗${NC} App is running but not responding"
        fi
    else
        echo -e "  ${YELLOW}○${NC} Stack not running"
    fi
    echo
}

test_health() {
    echo -e "${BLUE}[TEST] Running health checks...${NC}\n"
    
    if ! docker ps | grep -q rag-app-sfdc-pdf; then
        echo -e "${RED}✗ Container not running${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Checking API endpoints...${NC}"
    
    # RAG stats
    if RESULT=$(curl -s http://localhost:5000/api/rag/stats); then
        echo -e "${GREEN}✓${NC} /api/rag/stats: $RESULT"
    else
        echo -e "${RED}✗${NC} /api/rag/stats: Failed"
    fi
    
    echo -e "${GREEN}✓ All checks passed${NC}\n"
}

show_logs() {
    echo -e "${BLUE}[LOGS] Application logs${NC}\n"
    docker logs -f rag-app-sfdc-pdf --tail 50
}

push_image() {
    if [ -z "$1" ]; then
        echo -e "${RED}Registry required${NC}"
        echo "Usage: $0 push your-registry.azurecr.io"
        exit 1
    fi
    
    REGISTRY="$1"
    echo -e "${BLUE}[PUSH] Pushing image to $REGISTRY...${NC}"
    
    docker tag rag-app:1.0-sfdc-pdf $REGISTRY/rag-app:1.0-sfdc-pdf
    docker tag rag-app:1.0-sfdc-pdf $REGISTRY/rag-app:latest
    docker push $REGISTRY/rag-app:1.0-sfdc-pdf
    docker push $REGISTRY/rag-app:latest
    
    echo -e "${GREEN}✓ Image pushed to $REGISTRY${NC}\n"
}

stop_stack() {
    echo -e "${BLUE}[STOP] Stopping containers...${NC}"
    docker-compose -f docker-compose.sfdc-pdf.yml down
    echo -e "${GREEN}✓ Containers stopped${NC}\n"
}

clean_all() {
    echo -e "${BLUE}[CLEAN] Removing all containers and volumes...${NC}"
    read -p "This will DELETE all data. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f docker-compose.sfdc-pdf.yml down -v
        echo -e "${GREEN}✓ Cleanup complete${NC}\n"
    else
        echo -e "${YELLOW}Cleanup cancelled${NC}\n"
    fi
}

# Main
case "$ACTION" in
    build)
        build
        ;;
    deploy)
        deploy
        ;;
    status)
        status
        ;;
    test)
        test_health
        ;;
    logs)
        show_logs
        ;;
    push)
        push_image "$2"
        ;;
    stop)
        stop_stack
        ;;
    clean)
        clean_all
        ;;
    *)
        show_help
        ;;
esac
