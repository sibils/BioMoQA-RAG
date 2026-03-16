# BioMoQA RAG

Biomedical question answering API using hybrid retrieval (SIBILS + FAISS) and BioBERT extractive QA, with an optional Qwen3-8B generative fallback.

Running at: `http://sibils-prod-ai.lan.text-analytics.ch:9000`

---

## Architecture

```
Question
   ↓
Hybrid Retrieval (parallel)
├── SIBILS BM25 — concept-aware keyword search (medline + plazi)
└── FAISS dense — semantic search over local index
   ↓
Cross-Encoder Reranking  (ms-marco-MiniLM-L-6-v2)
   ↓
Relevance Filtering  → top 5 documents
   ↓
Extractive QA  (BioBERT — ktrapeznikov/biobert_v1.1_pubmed_squad_v2)
   └── if no confident span → Generative fallback (Qwen3-8B, vLLM, fp8)
   ↓
Structured JSON response
```

---

## API Endpoints

Interactive docs: `/docs`

### `POST /qa`

Answer a single question.

**Request:**
```json
{
  "question": "What causes malaria?",
  "mode": "hybrid",
  "retrieval_n": 50,
  "final_n": 5,
  "include_documents": false,
  "debug": false
}
```

`mode` options:
- `"hybrid"` (default) — BioBERT extractive span; Qwen3 fallback if not confident
- `"extractive"` — BioBERT only, never hallucinates
- `"generative"` — Qwen3 always

**Response:**
```json
{
  "sibils_version": "biomoqa-2.0",
  "success": true,
  "error": "",
  "question": "What causes malaria?",
  "collection": "medline+plazi",
  "model": "biobert",
  "ndocs_requested": 50,
  "ndocs_returned_by_SIBiLS": 47,
  "answers": [
    {
      "answer": "Malaria is caused by Plasmodium parasites transmitted by Anopheles mosquitoes.",
      "answer_score": 0.82,
      "docid": "12345678",
      "doc_retrieval_score": 0.91,
      "doc_text": "Title. Abstract...",
      "snippet_start": 14,
      "snippet_end": 87
    }
  ],
  "mode_used": "hybrid:extractive",
  "pipeline_time": 2.1
}
```

`answer_score` is the BioBERT span confidence (null for generative). `snippet_start`/`snippet_end` are character offsets into `doc_text` for passage highlighting.

### `GET /api/QA`

Backwards-compatible with `biodiversitypmc.sibils.org/api/QA`.

```
GET /api/QA?q=What+causes+malaria%3F&col=medline&n=5&mode=hybrid
```

### `POST /batch`

Answer multiple questions using extractive mode with parallel retrieval. Retrieval runs concurrently across all questions — a batch of N questions takes roughly the same wall time as a single question for the I/O-bound retrieval step.

**Request:**
```json
{
  "questions": [
    "What causes malaria?",
    "What diseases are associated with ticks?"
  ],
  "retrieval_n": 50,
  "final_n": 5,
  "col": null
}
```

**Response:**
```json
{
  "results": [ { ...same format as /qa... }, ... ],
  "count": 2
}
```

### `GET /health`

Returns pipeline status and loaded model info.

---

## Configuration

Edit [`config.toml`](config.toml) before deploying:

```toml
[model]
mode = "gpu"
model_name = "Qwen/Qwen3-8B"
gpu_memory_utilization = 0.83
quantization = "fp8"

[retrieval]
retrieval_n = 50

[reranking]
top_k = 15

[relevance_filter]
final_n = 5

[generation]
max_tokens = 512
temperature = 0.1
```

---

## Project Structure

```
BioMoQA-RAG/
├── src/biomoqa_rag/
│   ├── api_server.py           # FastAPI server + endpoints
│   ├── pipeline.py             # RAG pipeline (retrieval → QA → generation)
│   ├── config.py               # Config dataclasses (reads config.toml)
│   ├── build_dense_index.py    # Build FAISS index from seed queries
│   ├── retrieval/
│   │   ├── sibils_retriever.py # SIBILS BM25 API client
│   │   ├── dense_retriever.py  # FAISS semantic search
│   │   ├── parallel_hybrid.py  # Hybrid orchestration + RRF fusion
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   ├── relevance_filter.py # Overlap-based relevance filter
│   │   └── query_parser.py     # SIBILS query parser integration
│   └── extraction/
│       └── extractive_qa.py    # BioBERT span extraction
├── data/
│   ├── faiss_index.bin         # Dense index (rebuild with build_dense_index.py)
│   ├── documents.pkl           # ~2400 biomedical documents
│   ├── seed_queries.txt        # Seed queries for FAISS index construction
│   └── sibils_cache/           # Disk cache for SIBILS API responses (7-day TTL)
├── deploy/                     # Ansible deployment (see deploy/README.md)
├── results/                    # Evaluation outputs
├── config.toml                 # Runtime configuration
└── pyproject.toml              # Package + dependencies
```

---

## Deployment

Managed via Ansible. See [deploy/README.md](deploy/README.md) and [deploy/DEPLOY_GUIDE.md](deploy/DEPLOY_GUIDE.md).

```bash
cd deploy
ansible-playbook -i inventory.yml playbook.yml
```

The service runs as `sibils-qa.service` (systemd) on `sibils-prod-ai`.

### Rebuilding the FAISS index

After editing `data/seed_queries.txt`, rebuild the dense index:

```bash
python -m biomoqa_rag.build_dense_index
```

### GPU / MIG notes

The server runs on a GRID A100D-1-20C vGPU (MIG 1g.20GB, 20GB partition). The MIG UUID is auto-detected at startup via `nvidia-smi -L` — no manual configuration needed. See [deploy/DEPLOY_GUIDE.md](deploy/DEPLOY_GUIDE.md) for details.

---

## Troubleshooting

```bash
# Service status and logs
systemctl status sibils-qa
journalctl -u sibils-qa -f

# GPU usage
nvidia-smi

# Kill stale GPU process
nvidia-smi | grep python | awk '{print $5}' | xargs kill -9
```
