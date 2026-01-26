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

2. **Configure variables** in `group_vars/all.yml`:
   ```yaml
   # For CPU mode:
   biomoqa_mode: "cpu"
   biomoqa_model_size: "3b"  # or "1.5b" for less RAM

   # For GPU mode (default):
   biomoqa_mode: "gpu"
   ```

3. **Run the playbook:**
   ```bash
   ansible-playbook -i my-inventory.yml playbook.yml
   ```

## Configuration

All configuration is done via `config.toml` (generated from `group_vars/all.yml`).

### Inference Modes

| Mode | Variable | VRAM/RAM | Speed |
|------|----------|----------|-------|
| GPU (default) | `biomoqa_mode: "gpu"` | ~6GB VRAM | Fast |
| GPU small | `biomoqa_mode: "gpu_small"` | ~8GB VRAM | Fast |
| CPU 3B | `biomoqa_mode: "cpu", model_size: "3b"` | ~8GB RAM | Slow |
| CPU 1.5B | `biomoqa_mode: "cpu", model_size: "1.5b"` | ~4GB RAM | Medium |
| CPU 0.5B | `biomoqa_mode: "cpu", model_size: "0.5b"` | ~2GB RAM | Faster |

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

# Create config.toml (copy from repo and edit)
cp config.toml.example config.toml
# Edit config.toml as needed

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

## config.toml Reference

```toml
[server]
host = "0.0.0.0"
port = 9000
workers = 1
log_level = "info"

[model]
mode = "gpu"        # "gpu", "gpu_small", or "cpu"
size = "3b"         # "0.5b", "1.5b", "3b", "7b"
gpu_memory_utilization = 0.8
quantization = "fp8"

[generation]
max_tokens = 384
temperature = 0.1

[retrieval]
retrieval_n = 20
use_smart_retrieval = true
hybrid_alpha = 0.5

[reranking]
enabled = true
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k = 15

[relevance_filter]
enabled = true
min_overlap = 0.15
final_n = 10

[context]
max_abstract_length = 800
truncate_abstracts = true

[data]
faiss_index = "data/faiss_index.bin"
documents = "data/documents.pkl"
```
