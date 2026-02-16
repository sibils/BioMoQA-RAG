# Deployment Guide (sibils-prod-ai)

How to redeploy BioMoQA-RAG on the HULK VM (`sibils-prod-ai`).

## Infrastructure

- **VM**: `sibils-prod-ai.lan.text-analytics.ch`
- **GPU**: 20GB allocated (gpu_memory_utilization = 0.83)
- **Service**: `sibils-qa` (systemd)
- **API**: http://sibils-prod-ai.lan.text-analytics.ch:9000/docs
- **Ansible repo**: https://github.com/bitem-heg-geneve/ansible-cfg
- **Jenkins job**: https://jenkins.text-analytics.ch/job/Ansible/job/SIBiLS/job/SIBiLS_prod_qa_deploy/

## Step 1: Update config (if needed)

Edit the production config.toml in the Ansible repo:

https://github.com/bitem-heg-geneve/ansible-cfg/blob/main/files/playbook_prod_sibils_qa/config.toml

Commit and push changes to that repo.

## Step 2: Run Jenkins build

Go to the Jenkins deploy job and click **Build Now**:

https://jenkins.text-analytics.ch/job/Ansible/job/SIBiLS/job/SIBiLS_prod_qa_deploy/

This runs the Ansible playbook which:
- Pulls the latest BioMoQA-RAG code from GitHub
- Deploys the config.toml from the Ansible repo
- Restarts the systemd service

## Step 3: Verify on the VM

```bash
ssh esteban@sibils-prod-ai
sudo -s
systemctl status sibils-qa*
```

Check the logs if something goes wrong:

```bash
journalctl -fu sibils-qa
```

## Step 4: Test the API

```bash
# Health check
curl http://sibils-prod-ai.lan.text-analytics.ch:9000/health

# Ask a question
curl -X POST http://sibils-prod-ai.lan.text-analytics.ch:9000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes malaria?"}'
```

Or open the Swagger UI: http://sibils-prod-ai.lan.text-analytics.ch:9000/docs

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA spawn warning | `multiprocessing.set_start_method("spawn")` is set in api_server.py |
| Out of GPU memory | Lower `gpu_memory_utilization` in config.toml (current: 0.83) |
| Service keeps crashing | Check `journalctl -fu sibils-qa` for errors |
| Can't push to GitHub | Regenerate token: GitHub > Settings > Developer settings > Tokens |
