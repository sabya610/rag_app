#!/bin/bash

###############################################################################
# Package Helm Chart for Distribution
# Creates downloadable tar.gz with all Helm files
###############################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
HELM_CHART_DIR="./helm/rag-app-slack"
HELM_SUPPORT_FILES="./helm/*.yaml ./helm/*.sh ./helm/*.ps1 ./helm/*.md"
PACKAGE_DIR="./helm-releases"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="rag-app-slack-helm-${TIMESTAMP}.tar.gz"

echo -e "${BLUE}Creating Helm package for distribution...${NC}\n"

# Create package directory
mkdir -p "$PACKAGE_DIR"

# Package the Helm chart
echo -e "${YELLOW}Packaging Helm chart...${NC}"
helm package "$HELM_CHART_DIR" --destination "$PACKAGE_DIR"

# Find the generated package
CHART_PACKAGE=$(ls -t "$PACKAGE_DIR"/rag-app-slack-*.tgz | head -1)
CHART_PACKAGE_NAME=$(basename "$CHART_PACKAGE")

echo -e "${GREEN}✓ Chart packaged: $CHART_PACKAGE_NAME${NC}\n"

# Create deployment bundle
echo -e "${YELLOW}Creating deployment bundle...${NC}"

# Create temporary directory for bundle
BUNDLE_DIR="rag-app-slack-deployment-${TIMESTAMP}"
mkdir -p "$BUNDLE_DIR"

# Copy Helm chart
cp "$CHART_PACKAGE" "$BUNDLE_DIR/"

# Copy supporting files
cp ./helm/values-ftc-aie.yaml "$BUNDLE_DIR/" 2>/dev/null || true
cp ./helm/deploy-helm.sh "$BUNDLE_DIR/" 2>/dev/null || true
cp ./helm/Deploy-Helm.ps1 "$BUNDLE_DIR/" 2>/dev/null || true
cp ./helm/HELM_DEPLOYMENT_GUIDE.md "$BUNDLE_DIR/" 2>/dev/null || true
cp ./helm/QUICK_REFERENCE.md "$BUNDLE_DIR/" 2>/dev/null || true

# Create README for bundle
cat > "$BUNDLE_DIR/README.md" << 'EOF'
# RAG App Slack Integration - Helm Deployment Package

This package contains everything needed to deploy the RAG App with Slack Integration to a Kubernetes cluster.

## Contents

- `rag-app-slack-*.tgz` - Helm chart package
- `values-ftc-aie.yaml` - FTC AIE 1.1.1 cluster values
- `deploy-helm.sh` - Bash deployment script
- `Deploy-Helm.ps1` - PowerShell deployment script
- `HELM_DEPLOYMENT_GUIDE.md` - Comprehensive deployment guide
- `QUICK_REFERENCE.md` - Quick reference commands

## Quick Start

### Option 1: Automated Deployment (Windows)

```powershell
.\Deploy-Helm.ps1 -Action deploy
```

### Option 2: Automated Deployment (Linux/Mac)

```bash
chmod +x deploy-helm.sh
./deploy-helm.sh deploy
```

### Option 3: Manual Deployment

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm install rag-app-slack rag-app-slack-*.tgz \
  -f values-ftc-aie.yaml \
  --set slack.botToken=xoxb-YOUR_TOKEN \
  --set slack.signingSecret=YOUR_SECRET
```

## Prerequisites

1. Helm 3.x: https://helm.sh/docs/intro/install/
2. kubectl configured to access your cluster
3. Slack Bot Token (xoxb-...)
4. Slack Signing Secret

## Documentation

See `HELM_DEPLOYMENT_GUIDE.md` for comprehensive deployment instructions.

For quick reference, see `QUICK_REFERENCE.md`.

## Support

For issues or questions, refer to the deployment guide or check your cluster logs:

```bash
kubectl logs -n default -l app.kubernetes.io/instance=rag-app-slack -f
```

---

**Generated:** $(date)
**Version:** 1.0.0
EOF

# Make scripts executable
chmod +x "$BUNDLE_DIR/deploy-helm.sh" "$BUNDLE_DIR/Deploy-Helm.ps1" 2>/dev/null || true

# Create tar.gz package
echo -e "${YELLOW}Creating compressed archive...${NC}"
tar -czf "$PACKAGE_DIR/$PACKAGE_NAME" "$BUNDLE_DIR"

# Display results
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Helm Deployment Package Created${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${BLUE}Package Details:${NC}"
echo -e "  Location: $PACKAGE_DIR/$PACKAGE_NAME"
echo -e "  Size: $(du -h "$PACKAGE_DIR/$PACKAGE_NAME" | cut -f1)"
echo -e "  Contents: Helm chart + deployment scripts + documentation"

echo -e "\n${BLUE}To Use:${NC}"
echo "  1. Download: $PACKAGE_NAME"
echo "  2. Extract: tar -xzf $PACKAGE_NAME"
echo "  3. Read: README.md or HELM_DEPLOYMENT_GUIDE.md"
echo "  4. Deploy: ./Deploy-Helm.ps1 -Action deploy (Windows)"
echo "           or ./deploy-helm.sh deploy (Linux/Mac)"

echo -e "\n${BLUE}Chart Information:${NC}"
ls -lh "$CHART_PACKAGE"

echo -e "\n${BLUE}Bundle Contents:${NC}"
tar -tzf "$PACKAGE_DIR/$PACKAGE_NAME" | head -20

echo -e "\n${GREEN}✓ Ready for distribution!${NC}\n"

# Cleanup temporary directory
rm -rf "$BUNDLE_DIR"
