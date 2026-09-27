# SIBiLS QA deployment

Two environments, one repo. What an environment runs is decided by one line in its
playbook in [ansible-cfg](https://github.com/bitem-heg-geneve/ansible-cfg): `repo_version`.

| Host | Playbook | `repo_version` | Endpoints |
|---|---|---|---|
| `sibils-dev-ai` | `playbook_dev_sibils_qa.yml` | `master` | dev search + dev parser |
| `sibils-prod-ai` | `playbook_prod_sibils_qa.yml` | `prod_5.1.5.10` | prod search + prod parser |

## Why the branch is the whole story

`bitem.python_service_from_git` never clones onto the host. It builds a venv and runs
`pip install "git+git@github.com:sibils/BioMoQA-RAG.git@<repo_version>"`. No source tree
exists on the server, only what the wheel put in `site-packages/`, and the venv is rebuilt
from scratch whenever that commit changes.

So there is nothing to copy between environments and nothing to hand-edit on a host.
Change `repo_version`, run the playbook.

## Promoting dev to production

1. Let the change sit on `master` and be exercised on `sibils-dev-ai`.
2. Cut a release branch from the dev state you want:
   ```bash
   git fetch origin
   git push origin origin/master:refs/heads/prod_<version>
   ```
3. Change `repo_version` in `playbook_prod_sibils_qa.yml`.
4. Deploy prod from Jenkins.

Production never tracks `master`. A prod branch is a fixed point: redeploy it and you get
the same thing back. That is the entire reason for the split.

## Configuration

`config.toml` reaches the host through the role's `static_file_dir`, from
`files/<playbook name>/config.toml` in ansible-cfg. Do not edit it on the host; the next
deploy overwrites it.

**A setting you leave out of config.toml does not get a sensible value. It gets the
Pydantic default from `src/biomoqa_rag/config.py`, and several of those are development
defaults.** The one that matters:

```python
query_parser_api_url: str = "https://biodiversitypmc.dev.sibils.org/api/query/parse"
```

Omit that key from the production config and production calls the **dev** parser. The prod
config sets both `search_api_url` and `query_parser_api_url` explicitly for this reason.

Two settings are safe by default on `master` but are stated explicitly in the prod config
so they cannot drift silently:

- `empty_cache_ttl = 0` — never cache an empty result. A transient upstream zero would
  otherwise be replayed to every later request for the same query.
- `enforce_taxon_phrase = true` — require a recognised binomial as an adjacent phrase, so
  a specific epithet alone cannot match a different organism.

After changing `config.py`, check whether the deployed config.toml files still cover every
section that matters.

## GPU

`sibils-prod-ai` runs the A100 in MIG mode. In CUDA 12+, with `CUDA_VISIBLE_DEVICES`
unset, `cuda:0` resolves to the physical GPU, which is inaccessible under MIG, and vLLM
fails with `CUDA driver error: operation not supported`. Get the instance UUID with
`nvidia-smi -L` on the host; it changes if the partition is recreated.

The HuggingFace pre-download in `post_tasks` runs with `CUDA_VISIBLE_DEVICES=""` and
`PYTORCH_ALLOC_CONF` unset on purpose. Pulling those models into VRAM at that point caused
a GPU OOM at startup.

Memory budget for Qwen3-8B: ~8 GB of FP8 weights, plus ~144 KB/token of KV cache. At
`gpu_memory_utilization = 0.83` on a 20 GB partition there is room for
`max_model_len = 4096`. A 40 GB partition would support 16384.

## Verifying

```bash
curl http://sibils-dev-ai.lan.text-analytics.ch:9000/api/health
curl http://sibils-prod-ai.lan.text-analytics.ch:9000/api/health
```

Model load takes minutes, so the first check after a venv rebuild is slow.

| Symptom | Cause |
|---|---|
| `CUDA driver error: operation not supported` | `CUDA_VISIBLE_DEVICES` missing or stale MIG UUID |
| Garbage or CJK output | `model_name` is not a pre-quantized FP8 checkpoint while `quantization = "fp8"` |
| Empty results that persist | An empty upstream response got cached; `empty_cache_ttl` must be 0 |
| Prod behaving like dev | `query_parser_api_url` / `search_api_url` missing from the prod config |
| Out of GPU memory | Lower `gpu_memory_utilization`, or `max_model_len` too high for the partition |
| Config change had no effect | Edited on the host, or in this repo's mirror, instead of in ansible-cfg |
