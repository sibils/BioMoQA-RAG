# Handoff: bring up sibils-dev-ai

You are on **sibils-dev-ai.lan.text-analytics.ch**, a freshly provisioned VM that is meant
to become the new SIBiLS QA *development* host. Your job is to get its GPU working so the
Ansible deploy can complete. Nothing is broken for users while you do this.

## The one problem

A Jenkins run of `SIBiLS_dev_qa_deploy` (playbook `playbook_dev_sibils_qa.yml` from
`bitem-heg-geneve/ansible-cfg`) failed here:

```
TASK [bitem.python_service_from_git : Look for /dev/nvidiactl]
TASK [bitem.python_service_from_git : Fail if GPU is required but not found]
[ERROR]: NVIDIA GPU not detected: /dev/nvidiactl is missing.
        Make sure the driver is installed and the kernel module is loaded.
```

Everything before that succeeded: Python 3.13 installed, GitHub App token minted, all
argument validation passed. **The playbook and the config are not the problem.** The role's
check is a plain `stat /dev/nvidiactl`. Make `nvidia-smi` work on this host and the deploy
proceeds.

This was a first-time build (`[WS-CLEANUP]`, Python installed fresh), so treat the VM as
new: assume nothing has been set up.

## Diagnose in this order

The first step that fails tells you which layer is missing.

```bash
lspci | grep -i nvidia        # 1. is a GPU attached to this VM at all?
lsmod | grep -i nvidia        # 2. is the kernel module loaded?
ls -l /dev/nvidia*            # 3. do the device nodes exist?
nvidia-smi                    # 4. does the driver actually work?
```

| Result | Meaning | Action |
|---|---|---|
| `lspci` shows nothing | No GPU is attached to the VM | Hypervisor-side (Proxmox vGPU profile / passthrough). Not fixable from inside the guest. Stop and tell Esteban. |
| `lspci` shows it, `lsmod` empty | Driver missing or module not loading | Install the NVIDIA **vGPU guest driver** (see below) |
| Module loaded, no `/dev/nvidia*` | Nodes not created yet | `sudo nvidia-modprobe -c 0 -u`, or reboot |
| `nvidia-smi` works | Done | Re-run the Jenkins job |

## Driver version must match the host

These are **vGPU** profiles (`GRID A100D-*C`), not bare GPUs. The guest driver version has to
match the host's vGPU manager — a mismatched version installs fine and then fails to
initialise. Get the exact version from a host that already works:

```bash
# on the host currently serving qa.dev.sibils.org
nvidia-smi --query-gpu=driver_version --format=csv
cat /proc/driver/nvidia/version
```

Install the same version here. Do not pull a generic driver from the distro repo or from
NVIDIA's public download page; vGPU guest drivers come from the licensed bundle.

After installing, confirm all four diagnostic steps pass, then check the partition size:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## What this host should end up running

| | |
|---|---|
| Service | `sibils-qa`, installed at `/opt/sibils-qa` |
| Python | 3.13 (already installed by `bitem.install_python`) |
| Code | `pip install git+git@github.com:sibils/BioMoQA-RAG.git@master` |
| Model | `Qwen/Qwen3-8B-FP8`, vLLM, `gpu_memory_utilization = 0.83` |
| Port | 9000, health at `/api/health` |

The role does **not** clone onto the host — it builds a venv and pip-installs from the git
branch, so there is no source tree here. Do not edit anything under
`/opt/sibils-qa/venv/lib/python3.13/site-packages/biomoqa_rag/`; the next deploy rebuilds
the venv from scratch and discards it.

Likewise `config.toml` arrives from ansible-cfg via the role's `static_file_dir`. Editing
`/opt/sibils-qa/config.toml` on the host is pointless — the next deploy overwrites it.

## GPU memory, for when it comes up

Qwen3-8B-FP8 is ~8 GB of weights plus ~144 KB/token of KV cache. At
`gpu_memory_utilization = 0.83`, a 20 GB partition supports `max_model_len = 4096`. If this
VM has a smaller partition than the working host, vLLM will OOM at startup and that is the
first thing to check.

Two settings exist because of past GPU OOM incidents — leave them alone:

- `[extraction] device = -1` puts BioBERT on CPU. vLLM holds 83% of the GPU; putting BioBERT
  on it too causes CUDA OOM.
- The playbook's `post_tasks` pre-download HuggingFace models with `CUDA_VISIBLE_DEVICES=""`
  and `PYTORCH_ALLOC_CONF` unset. Pulling them into VRAM there caused an OOM at startup.

If the GPU turns out to be partitioned with MIG, the systemd unit needs
`CUDA_VISIBLE_DEVICES=MIG-<uuid>` (from `nvidia-smi -L`). Without it, CUDA 12 maps `cuda:0`
to the physical GPU, which is inaccessible under MIG, and vLLM dies with
`CUDA driver error: operation not supported`.

## Verifying once the deploy succeeds

```bash
systemctl status sibils-qa
journalctl -fu sibils-qa
curl http://localhost:9000/api/health
```

Model load takes minutes after a venv rebuild, so the first health check is slow. A healthy
response reports `"inference_mode": "gpu"` and `"use_vllm": true`. If it reports CPU, the
GPU is not being used and something above is still wrong.

## Do not

- Do not set `gpu: false` in the playbook to get past the error. That makes dev CPU-only and
  no longer representative of prod, which defeats the point of having this host.
- Do not deploy anything to `sibils-prod-ai` from this session.
- Do not push to `master` or `prod_5.1.5.10` in `sibils/BioMoQA-RAG`.

## Context you may need

- **Repo:** `github.com/sibils/BioMoQA-RAG` (the SIBiLS QA engine). `master` = dev,
  `prod_5.1.5.10` = production, cut from master at `13abd69` on 2026-09-27.
- **Ansible:** `github.com/bitem-heg-geneve/ansible-cfg`, playbooks
  `playbook_dev_sibils_qa.yml` and `playbook_prod_sibils_qa.yml`, run by Jenkins.
- **Open question for Esteban, not for you to decide:** the ansible-cfg commit that
  triggered this run was "Update repo_version to prod_5.1.5.10" but the *dev* job ran. The
  dev playbook must stay on `repo_version: master`; only the prod playbook uses
  `prod_5.1.5.10`. Worth confirming before a successful deploy puts the wrong branch here.
