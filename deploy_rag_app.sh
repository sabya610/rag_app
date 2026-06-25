#!/bin/bash
###############################################################################
# deploy_rag_app.sh — Deploy HPE Ezmeral RAG Knowledge Assistant to a K8s cluster
#
# Packages the Helm chart into a .tgz (or uses a pre-packaged one) and
# deploys the RAG app with all required configuration.
#
# Usage:
#   ./deploy_rag_app.sh [OPTIONS]
#
# Examples:
#   # Deploy from chart source (auto-packages first):
#   ./deploy_rag_app.sh --sid "YOUR_SID"
#
#   # Deploy from pre-packaged .tgz:
#   ./deploy_rag_app.sh --chart-pkg rag-app-0.2.0.tgz --sid "YOUR_SID"
#
#   # Full non-interactive:
#   ./deploy_rag_app.sh \
#     --namespace rag \
#     --sid "00Dd0000000bUlK!ARAA..." \
#     --proxy "http://10.79.90.173:443" \
#     --image-tag v2 \
#     --port-forward
#
#   # Package only (no deploy):
#   ./deploy_rag_app.sh --package-only
#
#   # Dry run:
#   ./deploy_rag_app.sh --dry-run --sid "TEST"
###############################################################################
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
NAMESPACE="rag"
RELEASE_NAME="rag-app"
IMAGE_REPO="sabya610/rag-app"
IMAGE_TAG="v2"
IMAGE_PULL_POLICY="Always"
CHART_DIR=""                    # auto-detected from script location
CHART_PKG=""                    # path to pre-packaged .tgz
CHART_PKG_DIR=""                # output dir for helm package
SF_SID=""
HTTP_PROXY=""
HTTPS_PROXY=""
NO_PROXY_EXTRA=""
GUNICORN_TIMEOUT=1800
PORT_FORWARD=false
PORT_FORWARD_ADDRESS="0.0.0.0"
LOCAL_PORT=80
DRY_RUN=false
SKIP_HELM=false
PACKAGE_ONLY=false

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[✗]${NC} $*" >&2; }
info()  { echo -e "${BLUE}[i]${NC} $*"; }
header(){ echo -e "\n${BLUE}━━━ $* ━━━${NC}"; }

# ─── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Deploy HPE Ezmeral RAG Knowledge Assistant to a Kubernetes cluster.

Chart Source (pick one):
      --chart-dir DIR        Path to Helm chart source directory (auto-packages)
      --chart-pkg FILE       Path to pre-packaged .tgz (skips packaging)
                             If neither given, auto-detects from script location.

Options:
  -n, --namespace NAME       Kubernetes namespace (default: rag)
  -r, --release NAME         Helm release name (default: rag-app)
  -t, --image-tag TAG        Docker image tag (default: v2)
  -s, --sid SID              Salesforce Session ID
  -p, --proxy URL            HTTP/HTTPS proxy URL (e.g. http://proxy:443)
      --no-proxy HOSTS       Additional NO_PROXY hosts (comma-separated)
      --gunicorn-timeout SEC Gunicorn worker timeout in seconds (default: 1800)
      --port-forward         Set up port-forward after deployment
      --port PORT            Local port for port-forward (default: 80)
      --package-only         Only package the chart, do not deploy
      --pkg-output DIR       Output dir for packaged chart (default: ./dist)
      --dry-run              Show what would be done without executing
      --skip-helm            Skip Helm install (only apply env/secrets)
  -h, --help                 Show this help

Prerequisites:
  - kubectl configured and pointing to target cluster
  - helm v3 installed
  - Cluster nodes need ≥8GB RAM for LLaMA 2 7B model
  - Internet access to pull sabya610/rag-app from Docker Hub
    (or pre-pull/mirror the image for air-gapped clusters)
EOF
    exit 0
}

# ─── Parse Arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--namespace)       NAMESPACE="$2"; shift 2 ;;
        -r|--release)         RELEASE_NAME="$2"; shift 2 ;;
        -t|--image-tag)       IMAGE_TAG="$2"; shift 2 ;;
        -s|--sid)             SF_SID="$2"; shift 2 ;;
        -p|--proxy)           HTTP_PROXY="$2"; HTTPS_PROXY="$2"; shift 2 ;;
        --no-proxy)           NO_PROXY_EXTRA="$2"; shift 2 ;;
        --chart-dir)          CHART_DIR="$2"; shift 2 ;;
        --chart-pkg)          CHART_PKG="$2"; shift 2 ;;
        --pkg-output)         CHART_PKG_DIR="$2"; shift 2 ;;
        --gunicorn-timeout)   GUNICORN_TIMEOUT="$2"; shift 2 ;;
        --port-forward)       PORT_FORWARD=true; shift ;;
        --port)               LOCAL_PORT="$2"; shift 2 ;;
        --package-only)       PACKAGE_ONLY=true; shift ;;
        --dry-run)            DRY_RUN=true; shift ;;
        --skip-helm)          SKIP_HELM=true; shift ;;
        -h|--help)            usage ;;
        *)                    err "Unknown option: $1"; usage ;;
    esac
done

# ─── Auto-detect chart source ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$CHART_PKG" && -z "$CHART_DIR" ]]; then
    # Look for chart source directory relative to script
    for candidate in \
        "$SCRIPT_DIR/helm/rag-app" \
        "$SCRIPT_DIR/rag-app" \
        "$SCRIPT_DIR" \
        "/root/rag-app-chart"; do
        if [[ -f "$candidate/Chart.yaml" ]]; then
            CHART_DIR="$candidate"
            break
        fi
    done

    # If no source dir found, look for a pre-packaged .tgz
    if [[ -z "$CHART_DIR" ]]; then
        for candidate in \
            "$SCRIPT_DIR"/rag-app-*.tgz \
            "$SCRIPT_DIR/helm"/rag-app-*.tgz \
            "$SCRIPT_DIR/dist"/rag-app-*.tgz; do
            if [[ -f "$candidate" ]]; then
                CHART_PKG="$candidate"
                break
            fi
        done
    fi

    if [[ -z "$CHART_DIR" && -z "$CHART_PKG" ]]; then
        err "Cannot find Helm chart source or .tgz package."
        err "Use --chart-dir or --chart-pkg to specify the location."
        exit 1
    fi
fi

# Default package output dir
if [[ -z "$CHART_PKG_DIR" ]]; then
    CHART_PKG_DIR="$SCRIPT_DIR/dist"
fi

# ─── Interactive prompts for missing values ──────────────────────────────────
if [[ -z "$SF_SID" ]]; then
    echo ""
    warn "No Salesforce Session ID provided."
    echo "  To get one: Log into https://hp.my.salesforce.com in your browser,"
    echo "  open DevTools → Application → Cookies → copy the 'sid' value."
    echo ""
    read -rp "Enter SF Session ID (or press Enter to skip SFDC): " SF_SID
fi

if [[ -z "$HTTP_PROXY" ]]; then
    read -rp "Enter HTTP proxy URL (or press Enter for none): " HTTP_PROXY
    HTTPS_PROXY="$HTTP_PROXY"
fi

# ─── Pre-flight checks ──────────────────────────────────────────────────────
header "Pre-flight Checks"

if ! command -v helm &>/dev/null; then
    err "helm not found. Install Helm v3: https://helm.sh/docs/intro/install/"
    exit 1
fi
log "helm: $(helm version --short 2>/dev/null)"

if [[ "$PACKAGE_ONLY" == "false" ]]; then
    if ! command -v kubectl &>/dev/null; then
        err "kubectl not found."
        exit 1
    fi
    log "kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client -o yaml 2>/dev/null | grep gitVersion | head -1)"

    if ! kubectl cluster-info &>/dev/null; then
        err "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    log "Cluster: $(kubectl cluster-info 2>/dev/null | head -1)"

    NODE_MEM=$(kubectl get nodes -o jsonpath='{.items[0].status.capacity.memory}' 2>/dev/null || echo "unknown")
    info "Node memory: $NODE_MEM"
fi

# ─── Step 1: Package Helm Chart ─────────────────────────────────────────────
header "Step 1: Package Helm Chart"

if [[ -n "$CHART_PKG" && -f "$CHART_PKG" ]]; then
    log "Using pre-packaged chart: $CHART_PKG"
    CHART_VERSION=$(echo "$CHART_PKG" | grep -oP 'rag-app-\K[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
elif [[ -n "$CHART_DIR" ]]; then
    CHART_VERSION=$(grep '^version:' "$CHART_DIR/Chart.yaml" | awk '{print $2}')
    info "Packaging chart v${CHART_VERSION} from: $CHART_DIR"

    # Lint the chart
    if [[ "$DRY_RUN" == "false" ]]; then
        helm lint "$CHART_DIR" || warn "Helm lint warnings (non-fatal)"
    fi

    # Create output directory and package
    mkdir -p "$CHART_PKG_DIR"
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] helm package $CHART_DIR --destination $CHART_PKG_DIR"
        CHART_PKG="$CHART_PKG_DIR/rag-app-${CHART_VERSION}.tgz"
    else
        HELM_PKG_OUTPUT=$(helm package "$CHART_DIR" --destination "$CHART_PKG_DIR" 2>&1)
        CHART_PKG=$(echo "$HELM_PKG_OUTPUT" | grep -oP '(?<=saved it to: ).*' || echo "$CHART_PKG_DIR/rag-app-${CHART_VERSION}.tgz")
        log "Packaged: $CHART_PKG ($(du -h "$CHART_PKG" | cut -f1))"
    fi
else
    err "No chart source or package found."
    exit 1
fi
info "Chart version: ${CHART_VERSION:-unknown}"

# ─── Package-only mode: stop here ───────────────────────────────────────────
if [[ "$PACKAGE_ONLY" == "true" ]]; then
    header "Package Complete"
    cat <<EOF

  Chart package: $CHART_PKG

  To deploy on another cluster, copy these 2 files and run:

    scp $CHART_PKG root@<MASTER_IP>:/root/
    scp $(readlink -f "$0") root@<MASTER_IP>:/root/

    ssh root@<MASTER_IP>
    chmod +x deploy_rag_app.sh
    ./deploy_rag_app.sh --chart-pkg /root/rag-app-${CHART_VERSION}.tgz \\
      --sid 'YOUR_SID' --proxy 'http://proxy:port' --port-forward

EOF
    exit 0
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
header "Deployment Configuration"
cat <<EOF
  Namespace:        $NAMESPACE
  Release:          $RELEASE_NAME
  Image:            $IMAGE_REPO:$IMAGE_TAG
  Pull Policy:      $IMAGE_PULL_POLICY
  Chart Package:    $CHART_PKG
  Chart Version:    ${CHART_VERSION:-unknown}
  SFDC SID:         ${SF_SID:+SET (${#SF_SID} chars)}${SF_SID:-NOT SET}
  HTTP Proxy:       ${HTTP_PROXY:-none}
  Gunicorn Timeout: ${GUNICORN_TIMEOUT}s
  Port Forward:     $PORT_FORWARD (port $LOCAL_PORT)
  Dry Run:          $DRY_RUN
EOF

if [[ "$DRY_RUN" == "true" ]]; then
    warn "DRY RUN mode — no changes will be made."
fi

echo ""
read -rp "Proceed with deployment? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ─── Helper: run or print ────────────────────────────────────────────────────
run() {
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] $*"
    else
        "$@"
    fi
}

# ─── Step 1: Create namespace ────────────────────────────────────────────────
header "Step 2: Create Namespace"
if kubectl get namespace "$NAMESPACE" &>/dev/null; then
    log "Namespace '$NAMESPACE' already exists"
else
    run kubectl create namespace "$NAMESPACE"
    log "Created namespace '$NAMESPACE'"
fi

# ─── Step 2: Helm install/upgrade ────────────────────────────────────────────
if [[ "$SKIP_HELM" == "false" ]]; then
    header "Step 3: Helm Install/Upgrade from Package"

    HELM_CMD=(helm upgrade --install "$RELEASE_NAME" "$CHART_PKG"
        --namespace "$NAMESPACE"
        --set "image.repository=$IMAGE_REPO"
        --set "image.tag=$IMAGE_TAG"
        --set "image.pullPolicy=$IMAGE_PULL_POLICY"
    )

    if [[ -n "$SF_SID" ]]; then
        HELM_CMD+=(--set "sfdc.sessionId=$SF_SID")
    fi

    info "Running: ${HELM_CMD[*]}"
    run "${HELM_CMD[@]}"
    log "Helm release '$RELEASE_NAME' deployed from $CHART_PKG"
else
    header "Step 3: Skipping Helm (--skip-helm)"
fi

# ─── Step 3: Wait for rollout ────────────────────────────────────────────────
header "Step 4: Wait for Deployment"
if [[ "$DRY_RUN" == "false" ]]; then
    info "Waiting for pods to be ready (timeout 5 min)..."
    kubectl rollout status deployment/"${RELEASE_NAME}-rag" \
        -n "$NAMESPACE" --timeout=300s
    log "Deployment is ready"
else
    info "[DRY RUN] Would wait for deployment/${RELEASE_NAME}-rag"
fi

# ─── Step 4: Apply proxy & Gunicorn env vars ─────────────────────────────────
header "Step 5: Apply Environment Variables"

ENV_ARGS=()
ENV_ARGS+=("GUNICORN_CMD_ARGS=--timeout=${GUNICORN_TIMEOUT}")
ENV_ARGS+=("GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT}")

if [[ -n "$HTTP_PROXY" ]]; then
    ENV_ARGS+=("HTTP_PROXY=$HTTP_PROXY")
    ENV_ARGS+=("HTTPS_PROXY=$HTTPS_PROXY")

    # Build NO_PROXY list
    ALL_NODE_IPS=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.addresses[?(@.type=="InternalIP")].address},{end}' 2>/dev/null | sed 's/,$//' || echo "")
    NO_PROXY="localhost,127.0.0.1,10.244.0.0/16,10.96.0.0/12,192.168.0.0/16,.cluster.local,.svc,${RELEASE_NAME}-postgres"
    [[ -n "$ALL_NODE_IPS" ]] && NO_PROXY="$NO_PROXY,$ALL_NODE_IPS"
    [[ -n "$NO_PROXY_EXTRA" ]] && NO_PROXY="$NO_PROXY,$NO_PROXY_EXTRA"
    ENV_ARGS+=("NO_PROXY=$NO_PROXY")

    info "Proxy: $HTTP_PROXY"
    info "NO_PROXY: $NO_PROXY"
fi

run kubectl set env deployment/"${RELEASE_NAME}-rag" -n "$NAMESPACE" "${ENV_ARGS[@]}"
log "Environment variables applied"

# ─── Step 5: Apply SFDC secret ───────────────────────────────────────────────
if [[ -n "$SF_SID" ]]; then
    header "Step 6: Apply SFDC Secret"
    run kubectl create secret generic "${RELEASE_NAME}-sfdc-secret" \
        -n "$NAMESPACE" \
        --from-literal="SF_SID=$SF_SID" \
        --from-literal=SFDC_CLIENT_ID='' \
        --from-literal=SFDC_CLIENT_SECRET='' \
        --from-literal=SFDC_USERNAME='' \
        --from-literal=SFDC_PASSWORD='' \
        --from-literal=SFDC_SECURITY_TOKEN='' \
        --dry-run=client -o yaml | kubectl apply -f -
    log "SFDC secret applied"
else
    warn "Step 6: Skipping SFDC secret (no SID provided)"
fi

# ─── Step 6: Wait for final rollout (env change triggers restart) ────────────
header "Step 7: Wait for Final Rollout"
if [[ "$DRY_RUN" == "false" ]]; then
    info "Waiting for rollout after env changes..."
    kubectl rollout status deployment/"${RELEASE_NAME}-rag" \
        -n "$NAMESPACE" --timeout=300s
    log "Final rollout complete"
fi

# ─── Step 7: Port-forward ────────────────────────────────────────────────────
if [[ "$PORT_FORWARD" == "true" && "$DRY_RUN" == "false" ]]; then
    header "Step 8: Port Forward"
    # Kill any existing port-forward on the local port
    fuser -k "${LOCAL_PORT}/tcp" 2>/dev/null || true
    sleep 2

    nohup kubectl port-forward svc/"${RELEASE_NAME}-service" \
        "${LOCAL_PORT}:80" -n "$NAMESPACE" \
        --address "$PORT_FORWARD_ADDRESS" > /tmp/rag-pf.log 2>&1 &
    sleep 3

    if grep -q "Forwarding" /tmp/rag-pf.log 2>/dev/null; then
        log "Port-forward active: http://${PORT_FORWARD_ADDRESS}:${LOCAL_PORT}"
    else
        warn "Port-forward may not be ready. Check: cat /tmp/rag-pf.log"
    fi
fi

# ─── Done ────────────────────────────────────────────────────────────────────
header "Deployment Complete"

POD_NAME=$(kubectl get pod -n "$NAMESPACE" --no-headers -o custom-columns=NAME:.metadata.name 2>/dev/null | grep "${RELEASE_NAME}-rag" | head -1)
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo "<node-ip>")

cat <<EOF

  ${GREEN}RAG App deployed successfully!${NC}

  Namespace:     $NAMESPACE
  Pod:           ${POD_NAME:-pending}
  Chart Package: $CHART_PKG

  Access the app:
    Port-forward: kubectl port-forward svc/${RELEASE_NAME}-service ${LOCAL_PORT}:80 -n ${NAMESPACE} --address 0.0.0.0
    Then open:    http://${NODE_IP}:${LOCAL_PORT}

  Verify SFDC:
    curl -s http://${NODE_IP}:${LOCAL_PORT}/sfdc/status

  View logs:
    kubectl logs deployment/${RELEASE_NAME}-rag -n ${NAMESPACE} -f

  Update SID later:
    kubectl create secret generic ${RELEASE_NAME}-sfdc-secret -n ${NAMESPACE} \\
      --from-literal=SF_SID='NEW_SID' \\
      --from-literal=SFDC_CLIENT_ID='' --from-literal=SFDC_CLIENT_SECRET='' \\
      --from-literal=SFDC_USERNAME='' --from-literal=SFDC_PASSWORD='' \\
      --from-literal=SFDC_SECURITY_TOKEN='' \\
      --dry-run=client -o yaml | kubectl apply -f -
    kubectl rollout restart deployment/${RELEASE_NAME}-rag -n ${NAMESPACE}

  Deploy to another cluster:
    scp $CHART_PKG deploy_rag_app.sh root@<NEW_MASTER>:/root/
    ssh root@<NEW_MASTER>
    chmod +x deploy_rag_app.sh
    ./deploy_rag_app.sh --chart-pkg /root/$(basename "$CHART_PKG") \\
      --sid 'YOUR_SID' --proxy 'http://proxy:port' --port-forward

EOF
