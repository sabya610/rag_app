#!/bin/bash
# ============================================================================
# HPE Ezmeral RAG Knowledge Assistant - K8s Deployment Script
# ============================================================================
# Usage:
#   ./deploy.sh                    # Deploy with defaults
#   ./deploy.sh --sid-file ~/sid.txt  # Deploy with SFDC session ID
#   ./deploy.sh --namespace myns   # Deploy to specific namespace
#   ./deploy.sh --uninstall        # Remove deployment
# ============================================================================

set -euo pipefail

# ---- Defaults ----
RELEASE_NAME="${RELEASE_NAME:-rag-app}"
NAMESPACE="${NAMESPACE:-default}"
CHART_DIR="$(cd "$(dirname "$0")/helm/rag-app" && pwd)"
IMAGE_REPO="${IMAGE_REPO:-sabya610/rag-app}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SID_FILE=""
UNINSTALL=false

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --sid-file)     SID_FILE="$2"; shift 2 ;;
    --namespace|-n) NAMESPACE="$2"; shift 2 ;;
    --release)      RELEASE_NAME="$2"; shift 2 ;;
    --image)        IMAGE_REPO="$2"; shift 2 ;;
    --tag)          IMAGE_TAG="$2"; shift 2 ;;
    --uninstall)    UNINSTALL=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---- Uninstall ----
if $UNINSTALL; then
  echo "[INFO] Uninstalling ${RELEASE_NAME} from namespace ${NAMESPACE}..."
  helm uninstall "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null || true
  echo "[OK] Uninstalled."
  exit 0
fi

# ---- Pre-flight checks ----
echo "============================================"
echo " RAG App K8s Deployment"
echo "============================================"
echo " Release:   ${RELEASE_NAME}"
echo " Namespace: ${NAMESPACE}"
echo " Image:     ${IMAGE_REPO}:${IMAGE_TAG}"
echo " Chart:     ${CHART_DIR}"
echo "============================================"

command -v helm >/dev/null 2>&1 || { echo "[ERROR] helm not found. Install helm first."; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "[ERROR] kubectl not found."; exit 1; }

# Create namespace if needed
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"

# ---- Build Helm overrides ----
HELM_ARGS=(
  upgrade --install "$RELEASE_NAME" "$CHART_DIR"
  --namespace "$NAMESPACE"
  --set "image.repository=${IMAGE_REPO}"
  --set "image.tag=${IMAGE_TAG}"
  --set "postgresql.enabled=true"
)

# If SID file provided, inject it into the SFDC secret
if [[ -n "$SID_FILE" && -f "$SID_FILE" ]]; then
  SID_VALUE=$(cat "$SID_FILE" | tr -d '\n\r')
  echo "[INFO] Using SFDC session ID from: ${SID_FILE}"
  HELM_ARGS+=(--set "sfdc.sessionId=${SID_VALUE}")
  HELM_ARGS+=(--set "env.SFDC_ENABLED=true")
else
  echo "[INFO] No SID file provided. SFDC features will use env vars or be disabled."
fi

# ---- Deploy ----
echo ""
echo "[INFO] Running: helm ${HELM_ARGS[*]}"
helm "${HELM_ARGS[@]}"

echo ""
echo "[OK] Deployment submitted. Checking status..."
echo ""

# ---- Wait & Status ----
echo "[INFO] Waiting for Postgres to be ready..."
kubectl rollout status deployment/"${RELEASE_NAME}-postgres" -n "$NAMESPACE" --timeout=120s 2>/dev/null || true

echo "[INFO] Waiting for RAG app to be ready..."
kubectl rollout status deployment/"${RELEASE_NAME}-rag" -n "$NAMESPACE" --timeout=300s 2>/dev/null || true

echo ""
echo "============================================"
echo " Deployment Complete"
echo "============================================"
kubectl get pods -n "$NAMESPACE" -l "app in (${RELEASE_NAME}-rag,${RELEASE_NAME}-postgres)"
echo ""

# ---- Access info ----
SVC_PORT=$(kubectl get svc "${RELEASE_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "80")
echo "Access the app:"
echo "  kubectl port-forward svc/${RELEASE_NAME}-service ${SVC_PORT}:${SVC_PORT} -n ${NAMESPACE}"
echo "  Then open: http://localhost:${SVC_PORT}"
echo ""
echo "To check logs:"
echo "  kubectl logs -f deployment/${RELEASE_NAME}-rag -n ${NAMESPACE}"
echo ""
echo "To update SID later:"
echo "  kubectl create secret generic ${RELEASE_NAME}-sfdc-secret --from-file=SF_SID=sid.txt -n ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -"
echo "  kubectl rollout restart deployment/${RELEASE_NAME}-rag -n ${NAMESPACE}"
