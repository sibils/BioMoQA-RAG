# BioMoQA RAG API Deployment Guide

## Current Status

✅ **V3.1 Fast API is RUNNING**
- **Address**: `0.0.0.0:9000` (externally accessible, NOT localhost!)
- **Speed**: 4.95s per question
- **Version**: V3.1 with FP8 quantization

---

## Network Access

### ✅ Already Accessible Externally!

The API is bound to `0.0.0.0:9000`, which means:

```
From inside VM:
  http://localhost:9000
  http://127.0.0.1:9000

From your network:
  http://172.30.120.7:9000

From internet (if firewall allows):
  http://<public-ip>:9000
```

**Test external access:**
```bash
# From another machine on your network
curl http://172.30.120.7:9000/health
```

---

## Why Systemd Service?

### Current Situation (Manual Start)

**Without systemd:**
```bash
# You run this:
./venv/bin/python3 -m uvicorn api_server_v3_fast:app --host 0.0.0.0 --port 9000

# Problems:
❌ Stops when you close terminal
❌ Stops when you logout/disconnect SSH
❌ Stops when VM reboots
❌ Doesn't restart if it crashes
❌ Need to manually start every time
```

### With Systemd Service

**After installing service:**
```bash
# API runs automatically, you don't need to do anything!

# Benefits:
✅ Runs FOREVER in background
✅ Survives terminal close / SSH disconnect
✅ AUTO-STARTS when VM reboots
✅ AUTO-RESTARTS if it crashes (every 10s)
✅ Proper logging to files
✅ Easy management with systemctl commands
✅ Professional production setup
```

**Think of it like:**
- MySQL, Nginx, PostgreSQL - they're always running, right?
- Systemd makes your API run the same way
- "Set it and forget it"

---

## Installing Systemd Service

### Option 1: Automatic Setup (Recommended)

```bash
cd /home/egaillac/BioMoQA-RAG
sudo ./setup_v3_fast_service.sh
```

This will:
1. Install service file to `/etc/systemd/system/`
2. Enable auto-start on boot
3. Start the service immediately
4. Show you the status

### Option 2: Manual Setup

```bash
# Copy service file
sudo cp biomoqa-rag-v3-fast.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable biomoqa-rag-v3-fast

# Start service now
sudo systemctl start biomoqa-rag-v3-fast

# Check status
sudo systemctl status biomoqa-rag-v3-fast
```

---

## Managing the Service

### Basic Commands

```bash
# Start the service
sudo systemctl start biomoqa-rag-v3-fast

# Stop the service
sudo systemctl stop biomoqa-rag-v3-fast

# Restart the service (e.g., after code update)
sudo systemctl restart biomoqa-rag-v3-fast

# Check status
sudo systemctl status biomoqa-rag-v3-fast

# View logs (live tail)
sudo journalctl -u biomoqa-rag-v3-fast -f

# View last 100 lines of logs
sudo journalctl -u biomoqa-rag-v3-fast -n 100

# Disable auto-start (if needed)
sudo systemctl disable biomoqa-rag-v3-fast
```

### Check If Running

```bash
# Method 1: systemctl
sudo systemctl status biomoqa-rag-v3-fast

# Method 2: Check port
ss -tlnp | grep 9000

# Method 3: Test API
curl http://localhost:9000/health
```

---

## Service Configuration

**Service file location:** `/etc/systemd/system/biomoqa-rag-v3-fast.service`

**Key settings:**
- `Restart=always` - Auto-restart on crash
- `RestartSec=10` - Wait 10s before restart
- `--host 0.0.0.0` - Bind to all interfaces (externally accessible)
- `--port 9000` - Port number

**Logs location:**
- Standard output: `/home/egaillac/BioMoQA-RAG/logs/api_v3_fast.log`
- Errors: `/home/egaillac/BioMoQA-RAG/logs/api_v3_fast_error.log`
- System logs: `journalctl -u biomoqa-rag-v3-fast`

---

## Accessing the API

### From Inside VM

```bash
# Health check
curl http://localhost:9000/health

# Ask a question
curl -X POST http://localhost:9000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes malaria?"}'

# Get retrieval info
curl http://localhost:9000/retrieval-info

# Compare versions
curl http://localhost:9000/compare
```

### From Your Network

```bash
# Replace with your actual VM IP
curl http://172.30.120.7:9000/health
```

### From Internet

**If you want internet access, you need to:**
1. Check firewall settings
2. Open port 9000 in cloud provider security group
3. Use public IP instead of private IP

---

## Production Checklist

### ✅ Current Setup

- [x] API running on 0.0.0.0:9000 (externally accessible)
- [x] V3.1 Fast pipeline (4.95s per question)
- [x] FP8 quantization enabled
- [x] Hybrid retrieval (SIBILS + Dense)
- [x] Service file created
- [x] Setup script ready

### ⏳ To Complete Production Setup

- [ ] Install systemd service: `sudo ./setup_v3_fast_service.sh`
- [ ] Test after reboot (service should auto-start)
- [ ] Set up monitoring (optional)
- [ ] Configure reverse proxy (optional, for HTTPS)
- [ ] Set up firewall rules (if needed)

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
sudo journalctl -u biomoqa-rag-v3-fast -n 50

# Check service status
sudo systemctl status biomoqa-rag-v3-fast

# Test manually first
cd /home/egaillac/BioMoQA-RAG
./venv/bin/python3 -m uvicorn api_server_v3_fast:app --host 0.0.0.0 --port 9000
```

### Port Already in Use

```bash
# Find what's using port 9000
sudo ss -tlnp | grep 9000

# Kill the process (if needed)
sudo kill <PID>

# Restart service
sudo systemctl restart biomoqa-rag-v3-fast
```

### GPU Memory Issues

```bash
# Clear GPU memory
nvidia-smi | grep python | awk '{print $5}' | xargs -r sudo kill -9

# Restart service
sudo systemctl restart biomoqa-rag-v3-fast
```

### Can't Access Externally

```bash
# Check if bound to 0.0.0.0 (not 127.0.0.1)
ss -tlnp | grep 9000
# Should show: 0.0.0.0:9000

# Check firewall
sudo ufw status

# Allow port 9000 (if needed)
sudo ufw allow 9000
```

---

## Comparison: Manual vs Systemd

| Feature | Manual Start | Systemd Service |
|---------|--------------|-----------------|
| **Persistence** | ❌ Stops on terminal close | ✅ Runs forever |
| **Auto-start** | ❌ Manual start needed | ✅ Auto-starts on boot |
| **Auto-restart** | ❌ Stays down if crashes | ✅ Restarts automatically |
| **Logging** | ❌ Lost when terminal closes | ✅ Permanent logs |
| **Management** | ❌ Complex process management | ✅ Simple systemctl commands |
| **Production** | ❌ Not suitable | ✅ Production-ready |

---

## Next Steps

### Recommended: Install Service Now

```bash
cd /home/egaillac/BioMoQA-RAG
sudo ./setup_v3_fast_service.sh
```

This will make your API:
- ✅ Always available
- ✅ Survive reboots
- ✅ Auto-recover from crashes
- ✅ Run like a professional service

### Optional Enhancements

1. **Reverse Proxy (Nginx)**
   - Add HTTPS
   - Load balancing
   - Better logging

2. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert on failures

3. **Rate Limiting**
   - Prevent abuse
   - Fair usage

4. **Authentication**
   - API keys
   - JWT tokens

---

## Summary

**Current Status:**
- ✅ API is running
- ✅ Externally accessible on 0.0.0.0:9000
- ✅ Fast (4.95s per question)
- ⏳ Not yet installed as permanent service

**Recommendation:**
Run `sudo ./setup_v3_fast_service.sh` to make it permanent and production-ready!

**Your VM IP:** `172.30.120.7`
**API URL:** `http://172.30.120.7:9000`
