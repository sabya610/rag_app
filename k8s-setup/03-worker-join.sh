#!/bin/bash
# ============================================================================
# K8s Worker Join - Run ONLY on worker (sabya-test-1)
# Expects the join command as argument or reads from stdin
# ============================================================================
set -euo pipefail

echo "============================================"
echo " K8s Worker Join - $(hostname)"
echo "============================================"

if [ $# -ge 1 ]; then
  JOIN_CMD="$*"
else
  echo "Usage: $0 <kubeadm join command>"
  echo "Example: $0 kubeadm join 10.227.104.247:6443 --token abc123 --discovery-token-ca-cert-hash sha256:xyz"
  exit 1
fi

echo "[1/1] Joining cluster..."
eval "$JOIN_CMD" --node-name="sabya-test-1"

echo ""
echo "============================================"
echo " Worker joined! Check on master with:"
echo "   kubectl get nodes"
echo "============================================"
