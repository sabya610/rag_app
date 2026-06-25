#!/bin/bash
# ============================================================================
# K8s Master Init - Run ONLY on master (sabya-test)
# ============================================================================
set -euo pipefail

MASTER_IP="10.227.104.247"
POD_CIDR="10.244.0.0/16"

echo "============================================"
echo " K8s Master Init - $(hostname)"
echo "============================================"

# ---- Initialize cluster ----
echo "[1/4] Initializing K8s control plane..."
kubeadm init \
  --apiserver-advertise-address="${MASTER_IP}" \
  --pod-network-cidr="${POD_CIDR}" \
  --node-name="sabya-test" \
  | tee /root/kubeadm-init.log

# ---- Configure kubectl for root ----
echo "[2/4] Setting up kubectl..."
mkdir -p /root/.kube
cp -f /etc/kubernetes/admin.conf /root/.kube/config
chown root:root /root/.kube/config
export KUBECONFIG=/root/.kube/config

# ---- Install Flannel CNI ----
echo "[3/4] Installing Flannel CNI..."
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

# ---- Generate join command ----
echo "[4/4] Generating worker join command..."
kubeadm token create --print-join-command > /root/k8s-join-command.sh
chmod +x /root/k8s-join-command.sh

echo ""
echo "============================================"
echo " Master setup complete!"
echo "============================================"
echo ""
echo "Join command saved to: /root/k8s-join-command.sh"
cat /root/k8s-join-command.sh
echo ""
echo "Waiting for nodes to be ready..."
sleep 10
kubectl get nodes
kubectl get pods -A
