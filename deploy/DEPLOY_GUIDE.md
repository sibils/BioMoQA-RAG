# Deployment Guide (sibils-prod-ai)

How to redeploy BioMoQA-RAG on the HULK VM (`sibils-prod-ai`).

## Infrastructure

- **VM**: `sibils-prod-ai.lan.text-analytics.ch`
- **GPU**: 20GB allocated (gpu_memory_utilization = 0.83)
- **Service**: `sibils-qa` (systemd)
- **API**: http://sibils-prod-ai.lan.text-analytics.ch:9000/docs
- **Jenkins job**: https://jenkins.text-analytics.ch/job/Ansible/job/SIBiLS/job/SIBiLS_prod_qa_deploy/

## Important: Two config.toml files

There are **two separate config management systems**. The production deployment
uses the **Ansible repo**, not the one in this repo.

| File | Purpose | Used in production? |
|------|---------|-------------------|
| **`ansible-cfg` repo**: [`files/playbook_prod_sibils_qa/config.toml`](https://github.com/bitem-heg-geneve/ansible-cfg/blob/main/files/playbook_prod_sibils_qa/config.toml) | Static config deployed by Jenkins/Ansible | **YES** |
| This repo: `config.toml` | Local development config | No (dev only) |
| This repo: `deploy/templates/config.toml.j2` | Jinja2 template (generic Ansible) | No (unused in production) |

When you change settings (e.g. `gpu_memory_utilization`), you must update
the config.toml in the **ansible-cfg repo**, not just this repo.

## Step 1: Push code changes

Make sure the latest code is pushed to this repo (sibils/BioMoQA-RAG):

```bash
git add -A && git commit -m "..." && git push
```

## Step 2: Update production config (if needed)

Edit the **static config.toml** in the Ansible repo:

https://github.com/bitem-heg-geneve/ansible-cfg/blob/main/files/playbook_prod_sibils_qa/config.toml

Make sure it matches the settings you want (e.g. `gpu_memory_utilization`,
`[sibils]` section with URLs and user_agent, etc.).

Commit and push to the ansible-cfg repo.

## Step 3: Run Jenkins build

Go to the Jenkins deploy job and click **Build Now**:

https://jenkins.text-analytics.ch/job/Ansible/job/SIBiLS/job/SIBiLS_prod_qa_deploy/

This runs the Ansible playbook which:
- Pulls the latest BioMoQA-RAG code from GitHub
- Deploys the config.toml from the ansible-cfg repo
- Restarts the systemd service

## Step 4: Verify on the VM

### Connect

```bash
ssh sibils-prod-ai.lan.text-analytics.ch   # connect to VM
sudo -s                                     # become root (enter your password)
```

### Service management

```bash
systemctl status sibils-qa*    # check if service is running
systemctl start sibils-qa      # start the service
systemctl stop sibils-qa       # stop the service
systemctl restart sibils-qa    # restart (e.g. after config change)
```

### Logs

```bash
journalctl -fu sibils-qa                        # follow live logs
journalctl -u sibils-qa --since "10 min ago"    # last 10 minutes only
```

### Switch to the service user

```bash
su sibils-qa -l    # switch to the sibils-qa system user (runs the service)
                   # useful to manually test commands as that user
```

## Step 5: Test the API

```bash
# Health check
curl http://sibils-prod-ai.lan.text-analytics.ch:9000/health

# Ask a question
curl -X POST http://sibils-prod-ai.lan.text-analytics.ch:9000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes malaria?"}'
```

Or open the Swagger UI: http://sibils-prod-ai.lan.text-analytics.ch:9000/docs

## GPU / MIG configuration

The VM uses an **NVIDIA A100 in MIG mode** (1g.20GB partition). In CUDA 12+, if
`CUDA_VISIBLE_DEVICES` is not set, `cuda:0` maps to the physical GPU which is
inaccessible in MIG mode → `RuntimeError: CUDA driver error: operation not supported`.

The MIG instance UUID (get it with `nvidia-smi -L`):
```
MIG-6238f33e-8323-510c-9388-bddb60a893d5
```

The systemd service file in the **ansible-cfg repo** must contain:
```ini
Environment="CUDA_VISIBLE_DEVICES=MIG-6238f33e-8323-510c-9388-bddb60a893d5"
```

To verify before redeploying:
```bash
CUDA_VISIBLE_DEVICES="MIG-6238f33e-8323-510c-9388-bddb60a893d5" \
  /opt/sibils-qa/venv/bin/python3 -c \
  "import torch; t = torch.zeros(1, device='cuda'); print('CUDA works:', t)"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA driver error: operation not supported` | Set `CUDA_VISIBLE_DEVICES=MIG-<UUID>` in service file (see above) |
| vLLM EngineCore subprocess fails | `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set in api_server.py |
| Out of GPU memory | Lower `gpu_memory_utilization` in config.toml (current: 0.83) |
| Service keeps crashing | Check `journalctl -fu sibils-qa` for errors |
| Config change not applied | Did you update the config.toml in **ansible-cfg repo** (not this repo)? |
| Can't push to GitHub | Regenerate token: GitHub > Settings > Developer settings > Tokens |
