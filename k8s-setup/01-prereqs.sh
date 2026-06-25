#!/bin/bash
# ============================================================================
# K8s Prerequisites - Run on ALL nodes (master + worker)
# Rocky Linux 8.x / RHEL 8.x
# ============================================================================
set -euo pipefail

MASTER_IP="10.227.104.247"
WORKER_IP="10.227.105.10"
POD_CIDR="10.244.0.0/16"
K8S_VERSION="1.30"

# ---- HPE Corporate Proxy ----
HTTP_PROXY="http://10.79.90.173:443"
HTTPS_PROXY="http://10.79.90.173:443"
NO_PROXY="localhost,127.0.0.1,${MASTER_IP},${WORKER_IP},${POD_CIDR},10.96.0.0/12,192.168.0.0/16,.cluster.local,.svc,.default.svc,.external.hpe.local,10.254.130.0/23,10.192.0.0/12,10.244.0.0/16,10.43.0.0/16,10.227.104.0/22"

export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export no_proxy="${NO_PROXY}"
export HTTP_PROXY HTTPS_PROXY NO_PROXY

echo "============================================"
echo " K8s Node Setup - $(hostname)"
echo " Proxy: ${HTTP_PROXY}"
echo "============================================"

# ---- 1. Set hostnames in /etc/hosts ----
echo "[1/9] Configuring /etc/hosts..."
grep -q "sabya-test " /etc/hosts || echo "${MASTER_IP}  sabya-test" >> /etc/hosts
grep -q "sabya-test-1" /etc/hosts || echo "${WORKER_IP}  sabya-test-1" >> /etc/hosts

# ---- 2. Disable swap ----
echo "[2/9] Disabling swap..."
swapoff -a
sed -i '/swap/d' /etc/fstab

# ---- 3. Disable SELinux ----
echo "[3/9] Setting SELinux to permissive..."
setenforce 0 || true
sed -i 's/^SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config

# ---- 4. Disable firewall ----
echo "[4/9] Disabling firewalld..."
systemctl stop firewalld 2>/dev/null || true
systemctl disable firewalld 2>/dev/null || true

# ---- 5. Load kernel modules + sysctl ----
echo "[5/9] Configuring kernel modules and sysctl..."
cat > /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF

modprobe overlay
modprobe br_netfilter

cat > /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system > /dev/null 2>&1

# ---- 6. Install containerd ----
echo "[6/9] Installing containerd..."
dnf install -y dnf-utils
dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
dnf install -y containerd.io

# Configure containerd with systemd cgroup
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

systemctl enable --now containerd
systemctl restart containerd

# ---- 7. Configure containerd proxy ----
echo "[7/9] Configuring containerd proxy..."
mkdir -p /etc/systemd/system/containerd.service.d
cat > /etc/systemd/system/containerd.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=${HTTP_PROXY}"
Environment="HTTPS_PROXY=${HTTPS_PROXY}"
Environment="NO_PROXY=${NO_PROXY}"
Environment="http_proxy=${HTTP_PROXY}"
Environment="https_proxy=${HTTPS_PROXY}"
Environment="no_proxy=${NO_PROXY}"
EOF

# Also configure kubelet proxy
mkdir -p /etc/systemd/system/kubelet.service.d
cat > /etc/systemd/system/kubelet.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=${HTTP_PROXY}"
Environment="HTTPS_PROXY=${HTTPS_PROXY}"
Environment="NO_PROXY=${NO_PROXY}"
Environment="http_proxy=${HTTP_PROXY}"
Environment="https_proxy=${HTTPS_PROXY}"
Environment="no_proxy=${NO_PROXY}"
EOF

systemctl daemon-reload
systemctl restart containerd

# ---- 8. Install kubeadm, kubelet, kubectl ----
echo "[8/9] Installing kubeadm, kubelet, kubectl (v${K8S_VERSION})..."
cat > /etc/yum.repos.d/kubernetes.repo <<EOF
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/rpm/repodata/repomd.xml.key
exclude=kubelet kubeadm kubectl cri-tools kubernetes-cni
EOF

dnf install -y kubelet kubeadm kubectl --disableexcludes=kubernetes
systemctl enable --now kubelet

# ---- 9. Pull K8s images ----
echo "[9/9] Pre-pulling K8s images..."
kubeadm config images pull

echo ""
echo "============================================"
echo " Node setup complete: $(hostname)"
echo " Next: Run 02-master-init.sh on master"
echo "============================================"
