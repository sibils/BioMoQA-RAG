# BioMoQA RAG Deployment

Ansible playbook for deploying BioMoQA RAG API on Ubuntu/Debian servers.

## Prerequisites

- Ansible 2.9+
- Target server: Ubuntu 20.04+ or Debian 11+
- SSH access to target server
- For GPU mode: NVIDIA GPU with 6GB+ VRAM and CUDA drivers installed

## Quick Start

1. **Copy and edit inventory:**
   ```bash
   cp inventory.yml my-inventory.yml
   # Edit my-inventory.yml with your server details
   ```

2. **Configure variables** in `group_vars/all.yml` or per-host:
   ```yaml
   # For CPU mode:
   biomoqa_use_cpu: true
   biomoqa_model_size: "3b"  # or "1.5b" for less RAM

   # For GPU mode (default):
   biomoqa_use_cpu: false
   ```

3. **Run the playbook:**
   ```bash
   ansible-playbook -i my-inventory.yml playbook.yml
   ```

## Configuration Options

### Inference Modes

| Mode | Variable | VRAM/RAM | Speed |
|------|----------|----------|-------|
| GPU (default) | `biomoqa_use_cpu: false` | ~6GB VRAM | Fast |
| GPU small | `biomoqa_use_gpu_small: true` | ~8GB VRAM | Fast |
| CPU 3B | `biomoqa_use_cpu: true, model_size: "3b"` | ~8GB RAM | Slow |
| CPU 1.5B | `biomoqa_use_cpu: true, model_size: "1.5b"` | ~4GB RAM | Medium |
| CPU 0.5B | `biomoqa_use_cpu: true, model_size: "0.5b"` | ~2GB RAM | Faster |

### Data Files

The FAISS index and documents need to be available. Options:

1. **Copy from local:** Set `biomoqa_copy_data: true` and `biomoqa_data_source`
2. **Symlink:** Set `biomoqa_link_data: true` and `biomoqa_data_source`
3. **Manual:** Copy files to `{{ biomoqa_install_dir }}/data/` after deployment

Required files:
- `data/faiss_index.bin`
- `data/documents.pkl`

## Service Management

```bash
# Status
sudo systemctl status biomoqa

# Logs
sudo journalctl -u biomoqa -f

# Restart
sudo systemctl restart biomoqa

# Stop
sudo systemctl stop biomoqa
```

## Environment Variables

All configuration is done via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `BIOMOQA_HOST` | 0.0.0.0 | Bind address |
| `BIOMOQA_PORT` | 9000 | Bind port |
| `BIOMOQA_WORKERS` | 1 | Uvicorn workers (keep 1 for GPU) |
| `BIOMOQA_LOG_LEVEL` | info | Log level |
| `BIOMOQA_USE_CPU` | false | Use CPU inference |
| `BIOMOQA_GPU_SMALL` | false | Use small GPU model |
| `BIOMOQA_MODEL_SIZE` | 3b | Model size (0.5b, 1.5b, 3b, 7b) |

## Manual Deployment (without Ansible)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project directory
mkdir -p /opt/biomoqa && cd /opt/biomoqa

# Initialize and install
uv init --no-readme
uv add git+ssh://git@github.com/sibils/BioMoQA-RAG.git
uv sync

# Copy data files
cp /path/to/faiss_index.bin /path/to/documents.pkl data/

# Create .env file
cat > .env << EOF
BIOMOQA_HOST=0.0.0.0
BIOMOQA_PORT=9000
BIOMOQA_USE_CPU=false
EOF

# Run
uv run biomoqa-api
```

## Health Check

```bash
curl http://localhost:9000/health
```

## API Usage

```bash
curl -X POST http://localhost:9000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes malaria?"}'
```
