# Pre-Deployment Checklist - Slack Integration on FTC AIE 1.1.1 Cluster

**Target Cluster:** FTC AIE 1.1.1  
**Deployment Branch:** `feature/slack-integration`  
**Deployment Type:** SFDC + PDF + Slack Integration  
**Deployment Host:** Installer (10.227.81.151)

---

## 1. ACCESS & CREDENTIALS ✓

### Cluster Access
- [x] Installer Host: `10.227.81.151` - **root/BDn@nPB42L!**
- [x] Coordinator: `10.227.81.154`
- [x] Master: `10.227.81.157`
- [x] Worker 1: `10.227.81.160`
- [x] Worker 2: `10.227.81.161`
- [x] Worker 3: `10.227.81.162`
- [x] NFS Host: `10.227.81.170`

### SSH Configuration
```bash
# Test SSH connectivity from Windows:
# Option 1: Using Windows Subsystem for Linux (WSL):
wsl ssh root@10.227.81.151

# Option 2: Using PuTTY or other SSH client
# Host: 10.227.81.151
# Port: 22
# Username: root
# Password: BDn@nPB42L!

# Option 3: Using PowerShell (if SSH module installed):
ssh root@10.227.81.151
```

---

## 2. SLACK APP SETUP ⚠ (REQUIRED)

Before deployment, you MUST create a Slack app and obtain credentials.

### Steps to Get Slack Credentials:

1. **Go to Slack API Dashboard:**
   - Navigate to: https://api.slack.com/apps
   - Click "Create New App" → "From scratch"
   - App Name: `rag-app-slack`
   - Workspace: Select your Slack workspace
   - Click "Create App"

2. **Get Bot Token:**
   - Left sidebar: Click "OAuth & Permissions"
   - Scroll to "Bot Token Scopes"
   - Add these scopes:
     - `channels:history` - Read channel messages
     - `channels:read` - Read channel info
     - `users:read` - Read user profiles
     - `chat:write` - Send messages (optional)
     - `groups:history` - Read private channel history
     - `groups:read` - Read private channel info
   - Click "Install to Workspace" at the top
   - Copy the **Bot User OAuth Token** (starts with `xoxb-`)
   - **Save this value for deployment**

3. **Get Signing Secret:**
   - Left sidebar: Click "Basic Information"
   - Scroll down to "App Credentials"
   - Copy **Signing Secret**
   - **Save this value for deployment**

4. **Request Access to Slack Workspace:**
   - Your app needs to be invited to channels to access them
   - In Slack, go to channel → Details → Integrations
   - Add your bot app

### Slack Credentials Checklist:
- [ ] Slack Bot Token: `xoxb-...` (starts with xoxb-)
- [ ] Slack Signing Secret: (32+ character string)
- [ ] Bot added to at least 1 Slack channel
- [ ] Channel ID(s) you want to import: e.g., `C1234567890`

---

## 3. SALESFORCE CREDENTIALS ⚠ (REQUIRED)

Obtain your Salesforce credentials if not already available:

### Steps to Get SFDC Credentials:

1. **SFDC Username & Password:**
   - Your Salesforce login credentials
   - Format: `user@company.com` / `password`

2. **Security Token:**
   - In Salesforce, click your avatar → Settings
   - Left sidebar: Personal → Reset My Security Token
   - Check your email for the token (32+ characters)

3. **Client ID & Secret (for OAuth):**
   - Optional if using direct login
   - Get from: Setup → Apps → App Manager → Connected Apps

### Salesforce Credentials Checklist:
- [ ] SFDC Username: (e.g., `user@company.com`)
- [ ] SFDC Password: (your SFDC password)
- [ ] SFDC Security Token: (sent via email)
- [ ] Client ID (optional): (if using OAuth)
- [ ] Client Secret (optional): (if using OAuth)

---

## 4. DOCKER HUB SETUP ⚠ (OPTIONAL)

Only needed if pushing images to Docker Hub for distribution.

### Setup Steps:

1. **Create Docker Hub Account (if not already):**
   - Go to: https://hub.docker.com
   - Sign up or log in

2. **Create Personal Access Token:**
   - Click your avatar → Account Settings
   - Left sidebar: Security → New Access Token
   - Name: `deployment-token`
   - Permissions: Read & Write
   - Copy the token

### Docker Hub Credentials Checklist:
- [ ] Docker Hub Username: (your Docker Hub username)
- [ ] Docker Hub Personal Access Token (or password): (your token)
- [ ] Repository created: `your-username/rag-app` (optional, auto-created)

### Set Environment Variables (PowerShell):
```powershell
$env:DOCKER_USERNAME = "your_docker_username"
$env:DOCKER_PASSWORD = "your_docker_access_token"
```

### Set Environment Variables (WSL/Linux):
```bash
export DOCKER_USERNAME="your_docker_username"
export DOCKER_PASSWORD="your_docker_access_token"
```

---

## 5. VERIFY LOCAL SETUP

### Windows PowerShell Requirements:
```powershell
# Check PowerShell version (5.0+)
$PSVersionTable.PSVersion

# Install SSH Module (if not present)
# Open PowerShell as Administrator and run:
# Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# Test SSH connectivity
Test-Connection -ComputerName 10.227.81.151 -Count 1
```

### WSL/Linux/Mac Requirements:
```bash
# Verify SSH is available
ssh -V

# Test connectivity
ping -c 1 10.227.81.151

# Test SSH login
ssh root@10.227.81.151 "echo 'SSH works'"
```

---

## 6. REPOSITORY VERIFICATION

### Ensure Slack Integration Branch is Ready:
```bash
cd c:\Users\malliks\rag_app

# Verify on feature/slack-integration branch
git branch -vv
git status

# Verify key files exist:
# - Dockerfile.slack-integrated
# - docker-compose.slack-integrated.yml
# - app/services/slack_client.py
# - app/services/slack_import.py
# - app/routes/slack_routes.py
# - init_slack_db.py
# - requirements.txt (with slack-sdk, slack-bolt)
```

### Verify Slack Dependencies in requirements.txt:
```bash
# Should contain:
# slack-sdk
# slack-bolt

grep -E "slack-sdk|slack-bolt" requirements.txt
```

---

## 7. DEPLOYMENT CONFIGURATION

### Option A: Using PowerShell Script (Windows)
```powershell
# Set environment variables first (optional)
$env:DOCKER_USERNAME = "your_username"
$env:DOCKER_PASSWORD = "your_password"

# Run the deployment script
cd c:\Users\malliks\rag_app
.\Deploy-ToCluster.ps1

# The script will prompt you for:
# 1. Slack Bot Token
# 2. Slack Signing Secret
# 3. SFDC Credentials (optional)
# 4. Docker Hub credentials (optional)
```

### Option B: Using Bash Script (Linux/Mac/WSL)
```bash
# Make script executable
chmod +x deploy-to-cluster.sh

# Set environment variables (optional)
export DOCKER_USERNAME="your_username"
export DOCKER_PASSWORD="your_password"

# Run the deployment
./deploy-to-cluster.sh

# The script will prompt for confirmation
```

### Option C: Manual SSH Deployment
```bash
# SSH to Installer host
ssh root@10.227.81.151

# Clone/update repository
mkdir -p /opt/rag_app
cd /opt/rag_app
git clone -b feature/slack-integration https://github.com/sabya610/rag_app.git .

# Create .env file with your credentials
nano .env

# Build Docker image
docker build -f Dockerfile.slack-integrated -t rag-app:slack-integrated .

# (Optional) Push to Docker Hub
docker login
docker tag rag-app:slack-integrated your-username/rag-app:slack-integrated
docker push your-username/rag-app:slack-integrated

# Deploy stack
docker-compose -f docker-compose.slack-integrated.yml up -d

# Initialize Slack database
docker-compose -f docker-compose.slack-integrated.yml exec rag-app-slack-integrated python init_slack_db.py

# Check status
docker-compose -f docker-compose.slack-integrated.yml ps
```

---

## 8. POST-DEPLOYMENT VALIDATION

### Test Endpoints:
```bash
# From Installer host or any host with network access to 10.227.81.151:

# SFDC+PDF stats
curl http://10.227.81.151:5000/api/rag/stats

# Slack stats
curl http://10.227.81.151:5001/api/slack/stats

# Slack channels
curl http://10.227.81.151:5001/api/slack/channels

# Test Slack import
curl -X POST http://10.227.81.151:5001/api/slack/import \
  -H "Content-Type: application/json" \
  -d '{
    "channel_ids": ["C1234567890"],
    "days_back": 7
  }'

# Test Slack search
curl -X POST http://10.227.81.151:5001/api/slack/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test",
    "search_type": "semantic",
    "limit": 10
  }'
```

### Check Container Logs:
```bash
ssh root@10.227.81.151 << 'EOF'
cd /opt/rag_app

# View all container status
docker-compose -f docker-compose.slack-integrated.yml ps

# View RAG app logs
docker-compose -f docker-compose.slack-integrated.yml logs -f rag-app-slack-integrated

# View PostgreSQL logs
docker-compose -f docker-compose.slack-integrated.yml logs postgres-slack-integrated

# Check database
docker-compose -f docker-compose.slack-integrated.yml exec postgres-slack-integrated psql -U postgres -d ragdb_slack -c "\dt"
EOF
```

---

## 9. ROLLBACK PLAN

### If Issues Occur:
```bash
# Stop Slack stack
ssh root@10.227.81.151 << 'EOF'
cd /opt/rag_app
docker-compose -f docker-compose.slack-integrated.yml down

# Optional: Remove volumes to clean up
docker-compose -f docker-compose.slack-integrated.yml down -v
EOF

# Revert to previous state:
# - Switch git branch: git checkout main
# - Redeploy SFDC+PDF only
# - All SFDC data remains in separate database
```

### Backup Slack Data:
```bash
ssh root@10.227.81.151 << 'EOF'
cd /opt/rag_app

# Backup Slack database
docker-compose -f docker-compose.slack-integrated.yml exec postgres-slack-integrated \
  pg_dump -U postgres ragdb_slack > slack_db_backup_$(date +%Y%m%d).sql

# Backup entire docker volume
docker run --rm -v postgres_slack_integrated_data:/data -v $(pwd):/backup \
  busybox tar czf /backup/slack_db_volume_$(date +%Y%m%d).tar.gz -C /data .
EOF
```

---

## 10. DEPLOYMENT TIMELINE

| Step | Estimated Time | Notes |
|------|----------------|-------|
| Prerequisites install | 5-10 min | One-time only |
| Repository clone/update | 2-3 min | First time longer |
| Docker build | 10-15 min | Depends on internet speed |
| Docker push (optional) | 5-10 min | Only if using Docker Hub |
| Stack deployment | 2-3 min | Containers start |
| Database initialization | 3-5 min | Slack tables created |
| Testing | 2-3 min | Health checks |
| **TOTAL** | **~30-50 min** | Includes wait times |

---

## 11. QUICK REFERENCE

### Slack Integration Ports:
- **SFDC+PDF API:** Port 5000
- **Slack Integration API:** Port 5001
- **PostgreSQL (SFDC):** Port 5432
- **PostgreSQL (Slack):** Port 5433

### Key Files:
- Deployment script (bash): `deploy-to-cluster.sh`
- Deployment script (PS): `Deploy-ToCluster.ps1`
- Docker image: `Dockerfile.slack-integrated`
- Docker compose: `docker-compose.slack-integrated.yml`
- Slack models: `app/models.py` (SlackMessage, SlackThread)
- Slack routes: `app/routes/slack_routes.py`
- Database init: `init_slack_db.py`

### Useful Commands:
```bash
# SSH to Installer
ssh root@10.227.81.151

# Monitor deployment
docker-compose -f /opt/rag_app/docker-compose.slack-integrated.yml logs -f

# Check resource usage
docker stats

# Restart specific service
docker-compose restart rag-app-slack-integrated

# Execute command in container
docker-compose exec rag-app-slack-integrated python -c "import slack_sdk; print('OK')"
```

---

## 12. SUPPORT & TROUBLESHOOTING

### Common Issues:

**Issue:** "Connection refused" on Slack endpoints
- **Fix:** Check if container is running: `docker ps | grep rag-app`
- **Fix:** Check logs: `docker logs rag-app-slack-integrated`

**Issue:** "Database connection error"
- **Fix:** Verify PostgreSQL is running: `docker-compose ps postgres-slack-integrated`
- **Fix:** Check .env file has correct DB_HOST, DB_USER, DB_PASS

**Issue:** "Slack token invalid"
- **Fix:** Verify token format: `xoxb-...` (minimum 100+ characters)
- **Fix:** Verify token is for correct Slack workspace
- **Fix:** Verify token hasn't expired or been revoked

**Issue:** "Docker image build fails"
- **Fix:** Check internet connectivity
- **Fix:** Check Python dependencies in requirements.txt
- **Fix:** Try: `docker build --no-cache -f Dockerfile.slack-integrated .`

### Emergency Contacts:
- Repository: https://github.com/sabya610/rag_app
- Slack API Docs: https://api.slack.com/docs
- Docker Docs: https://docs.docker.com
- PostgreSQL Docs: https://www.postgresql.org/docs

---

## Ready to Deploy?

✓ Print/bookmark this checklist  
✓ Gather all credentials  
✓ Verify SSH access to 10.227.81.151  
✓ Run deployment script from Windows PowerShell or WSL  
✓ Monitor logs during deployment  
✓ Test endpoints after deployment  
✓ Keep backup of database  

**Estimated Duration:** 30-50 minutes including prerequisites

---

**Last Updated:** 2026-06-26
**Version:** 1.0
