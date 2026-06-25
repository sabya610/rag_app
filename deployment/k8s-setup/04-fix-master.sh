#!/bin/bash
# ============================================================================
# Fix K8s Master - Reset and re-init with proper proxy exclusions
# Run on master (sabya-test) only
# ============================================================================
set -euo pipefail

MASTER_IP="10.227.104.247"
WORKER_IP="10.227.105.10"
POD_CIDR="10.244.0.0/16"

HP_PROXY="http://10.79.90.173:443"
NO_PROXY_VAL="localhost,127.0.0.1,10.227.104.247,10.227.105.10,10.244.0.0/16,10.96.0.0/12,192.168.0.0/16,.cluster.local,.svc,.default.svc,10.227.104.0/22,10.192.0.0/12,10.43.0.0/16"

echo "============================================"
echo " K8s Master Fix - Proxy + Reset + Re-init"
echo "============================================"

# ---- Step 1: Set proxy globally in /etc/environment ----
echo "[1/6] Setting proxy in /etc/environment..."
cat > /etc/environment <<EOF
http_proxy=${HP_PROXY}
https_proxy=${HP_PROXY}
HTTP_PROXY=${HP_PROXY}
HTTPS_PROXY=${HP_PROXY}
no_proxy=${NO_PROXY_VAL}
NO_PROXY=${NO_PROXY_VAL}
EOF

# ---- Step 2: Set proxy in shell profile ----
echo "[2/6] Setting proxy in /etc/profile.d/proxy.sh..."
cat > /etc/profile.d/proxy.sh <<EOF
export http_proxy="${HP_PROXY}"
export https_proxy="${HP_PROXY}"
export HTTP_PROXY="${HP_PROXY}"
export HTTPS_PROXY="${HP_PROXY}"
export no_proxy="${NO_PROXY_VAL}"
export NO_PROXY="${NO_PROXY_VAL}"
EOF
chmod +x /etc/profile.d/proxy.sh

# Source it now
export http_proxy="${HP_PROXY}"
export https_proxy="${HP_PROXY}"
export HTTP_PROXY="${HP_PROXY}"
export HTTPS_PROXY="${HP_PROXY}"
export no_proxy="${NO_PROXY_VAL}"
export NO_PROXY="${NO_PROXY_VAL}"

# ---- Step 3: Update containerd + kubelet systemd proxy ----
echo "[3/6] Updating systemd proxy for containerd and kubelet..."

mkdir -p /etc/systemd/system/containerd.service.d
cat > /etc/systemd/system/containerd.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=${HP_PROXY}"
Environment="HTTPS_PROXY=${HP_PROXY}"
Environment="NO_PROXY=${NO_PROXY_VAL}"
Environment="http_proxy=${HP_PROXY}"
Environment="https_proxy=${HP_PROXY}"
Environment="no_proxy=${NO_PROXY_VAL}"
EOF

mkdir -p /etc/systemd/system/kubelet.service.d
cat > /etc/systemd/system/kubelet.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=${HP_PROXY}"
Environment="HTTPS_PROXY=${HP_PROXY}"
Environment="NO_PROXY=${NO_PROXY_VAL}"
Environment="http_proxy=${HP_PROXY}"
Environment="https_proxy=${HP_PROXY}"
Environment="no_proxy=${NO_PROXY_VAL}"
EOF

systemctl daemon-reload
systemctl restart containerd
systemctl restart kubelet

# ---- Step 4: Reset previous kubeadm attempt ----
echo "[4/6] Resetting previous kubeadm init..."
kubeadm reset -f --cri-socket unix:///var/run/containerd/containerd.sock
rm -rf /etc/cni/net.d /var/lib/etcd /root/.kube
iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X

# ---- Step 5: Re-init cluster ----
echo "[5/6] Re-initializing K8s control plane..."
kubeadm init \
  --apiserver-advertise-address="${MASTER_IP}" \
  --pod-network-cidr="${POD_CIDR}" \
  --node-name="sabya-test" \
  --v=5 \
  | tee /root/kubeadm-init.log

# ---- Step 6: Post-init setup ----
echo "[6/6] Post-init setup..."
mkdir -p /root/.kube
cp -f /etc/kubernetes/admin.conf /root/.kube/config
chown root:root /root/.kube/config
export KUBECONFIG=/root/.kube/config

# Install Flannel CNI
echo "Installing Flannel CNI..."
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

# Generate join command
echo "Generating worker join command..."
kubeadm token create --print-join-command > /root/k8s-join-command.sh
chmod +x /root/k8s-join-command.sh

echo ""
echo "============================================"
echo " Master re-init complete!"
echo "============================================"
echo "Join command:"
cat /root/k8s-join-command.sh
echo ""
sleep 15
echo "Cluster status:"
kubectl get nodes
kubectl get pods -A
