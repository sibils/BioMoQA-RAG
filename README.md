# BioMoQA RAG

Biomedical question-answering API for SIBILS. Supports extractive (BioBERT span) and generative (Qwen3-8B) answer modes with BM25 or hybrid dense retrieval over Medline, Plazi, and PMC.

Running at: `http://sibils-prod-ai.lan.text-analytics.ch:9000`

---

## Architecture

```
User question
    │
    ├─ retrieval=sparse ──► SIBILSRetriever (BM25, disk-cached)
    │
    └─ retrieval=dense  ──► SmartHybridRetriever
                               ├─ SIBILS BM25 (n=10)  ┐
                               └─ FAISS dense  (k=10)  ┤── RRF fusion (alpha=0.5)
                                   S-PubMedBert-MSMARCO │
                                                        ▼
                                              CrossEncoder reranker
                                              (ms-marco-MiniLM-L-12-v2)
                                              + top-5 cap
    │
    ├─ mode=extractive ──► BioBERT single-pass (ktrapeznikov/biobert_v1.1_pubmed_squad_v2)
    └─ mode=generative ──► Qwen3-8B-FP8 via vLLM
```

**Recommended pairings:**
- Extractive → `sparse` (BM25 precision optimal for span extraction)
- Generative → `dense` (semantic coverage for synthesis)

---

## API

Interactive docs: `/docs`

### `POST /qa/multi` — main endpoint

Answer a question across all collections (medline, plazi, pmc), ranked best-first.

**Request:**
```json
{
  "question": "What causes malaria?",
  "mode": "extractive",
  "retrieval": "sparse"
}
```

| Field | Values | Default |
|-------|--------|---------|
| `mode` | `"extractive"` · `"generative"` | `"generative"` |
| `retrieval` | `"sparse"` (BM25 only) · `"dense"` (FAISS+BM25+reranker) | `"sparse"` |
| `retrieval_n` | int | config default (10) |
| `final_n` | int | config default (5) |

**Response:**
```json
{
  "question": "What causes malaria?",
  "mode_used": "extractive",
  "ndocs_retrieved": 5,
  "model": "biobert",
  "pipeline_time": 0.4,
  "collection_results": [
    {
      "collection": "medline",
      "rank": 1,
      "answers": [
        {
          "answer": "Plasmodium parasites transmitted by Anopheles mosquitoes",
          "answer_score": 0.82,
          "docs": [
            {
              "docid": "12345678",
              "doc_source": "medline",
              "doc_retrieval_score": 0.91,
              "doc_text": "...",
              "snippet_start": 14,
              "snippet_end": 70
            }
          ]
        }
      ]
    }
  ]
}
```

**Answer shape:**
- **Extractive**: `docs` has 1 element; `snippet_start`/`snippet_end` are char offsets into `doc_text`; `answer_score` is BioBERT confidence
- **Generative**: `docs` has N cited elements; `snippet_start`/`snippet_end` are `null`; `answer_score` is `null`

### `POST /qa`

Same as `/qa/multi` — kept for backwards compatibility.

### `GET /api/QA`

Backwards-compatible with `biodiversitypmc.sibils.org/api/QA`.

```
GET /api/QA?q=What+causes+malaria%3F&col=medline&n=5&mode=extractive
```

### `POST /batch`

Extractive QA over a list of questions with parallel retrieval.

```json
{
  "questions": ["What causes malaria?", "What diseases are associated with ticks?"],
  "retrieval_n": 10,
  "final_n": 5
}
```

### `GET /health`

Pipeline status, loaded models, and index size.

---

## Configuration

[`config.toml`](config.toml) — the key sections:

```toml
[model]
mode = "gpu"
model_name = "Qwen/Qwen3-8B-FP8"
gpu_memory_utilization = 0.83
quantization = "fp8"

[retrieval]
retrieval_n = 10

[sibils]
collections = ["medline", "plazi", "pmc"]
cache_dir = "data/sibils_cache"

[reranking]
model = "cross-encoder/ms-marco-MiniLM-L-12-v2"

[relevance_filter]
final_n = 5
```

---

## Project structure

```
BioMoQA-RAG/
├── src/biomoqa_rag/
│   ├── api_server.py           # FastAPI server + endpoints
│   ├── pipeline.py             # RAG pipeline orchestration
│   ├── config.py               # Config dataclasses (reads config.toml)
│   ├── retrieval/
│   │   ├── sibils_retriever.py # SIBILS BM25 API client (disk-cached)
│   │   ├── dense_retriever.py  # FAISS semantic search
│   │   ├── parallel_hybrid.py  # RRF fusion + SmartHybridRetriever
│   │   ├── reranker.py         # CrossEncoder reranker
│   │   ├── relevance_filter.py # Token-overlap relevance filter
│   │   └── query_parser.py     # SIBILS query parser integration
│   └── extraction/
│       └── extractive_qa.py    # BioBERT span extraction
├── eval/                       # Evaluation scripts (run from repo root)
│   ├── evaluate_comparison.py  # Old system vs new extractive pipeline
│   ├── evaluate_retrieval.py   # sparse vs dense retrieval quality
│   ├── evaluate_alpha.py       # RRF alpha + embedding model sweep
│   ├── evaluate_bioasq_qa.py   # QA-only benchmark (gold context)
│   └── evaluate_semantic.py    # BERTScore + cosine similarity eval
├── data/
│   ├── faiss_index.bin         # FAISS index (managed via ansible, not git)
│   ├── documents.pkl           # Indexed documents (managed via ansible)
│   └── sibils_cache/           # SIBILS API disk cache (7-day TTL)
├── results/                    # Evaluation outputs
├── rebuild_faiss_from_cache.py # Rebuild FAISS index from SIBILS cache
├── config.toml                 # Runtime configuration
└── pyproject.toml              # Package + dependencies
```

---

## Deployment

Managed via Ansible/Jenkins. The service runs as `sibils-qa.service` on `sibils-prod-ai`.

HuggingFace models are pre-downloaded to `/opt/sibils-qa/.cache/huggingface/` during deploy. The FAISS index (`data/faiss_index.bin`, `data/documents.pkl`) is deployed via the ansible `static_file_dir`.

### Rebuilding the FAISS index

```bash
python rebuild_faiss_from_cache.py
```

Reads the SIBILS disk cache (`data/sibils_cache/cache`), deduplicates by PMID, re-embeds with `pritamdeka/S-PubMedBert-MS-MARCO`, and writes a new `data/faiss_index.bin` + `data/documents.pkl`. Then upload both files to the ansible static dir.

---

## Troubleshooting

```bash
# Service status and logs
systemctl status sibils-qa
journalctl -u sibils-qa -f

# GPU usage
nvidia-smi
```
