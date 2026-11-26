# BioMoQA RAG

Biomedical question-answering API for SIBILS, over Medline, Plazi and PMC. Each question is
searched at a chosen depth (`fast` or `deep`), the five best documents are kept by a
cross-encoder, and they are read by a language model on the shared GPU server (generative) or
by BioBERT (extractive).

Demo: [qa.dev.sibils.org](https://qa.dev.sibils.org)

---

## Architecture

```
User question
    │
    ├─ doc_refs provided ──► DocResolver (PMID / PMCID / DOI / title → Document)
    │                              └─ skips retrieval entirely
    │
    └─ per collection, at the depth asked (depth.py)
         fast: query parser's query (species filter) ─┐
               plain OR search of the question's words ┤── 2 × 20 candidates
         deep: + two searches written by the language model,
                 + one on its guess when it is sure (kept only if they match the question)
                                                       ▼
                                  CrossEncoder reranker (ms-marco-MiniLM-L-12-v2)
                                  every candidate scored against the question as asked
                                                       ▼
                                               the five best documents
    │
    ├─ mode=extractive ──► BioBERT (ktrapeznikov/biobert_v1.1_pubmed_squad_v2)
    └─ mode=generative ──► the shared language-model server (Qwen3.8-27B, llm_remote.py)
                           no first token in 4 s ──► the extractive answer, with a note
```

The searches are keyword searches only. The old dense path (a FAISS index of 4,739 documents from
the service's own search cache) was removed in November 2026. A dense first stage over the whole
corpus is to be tried again when more GPU is available: it needs every abstract embedded, and on
the test questions a 2023 encoder (MedCPT) added little beside the two keyword searches.

---

## API

Interactive docs: `/api/` on the service host (FastAPI, generated from the code).

### `POST /qa` — main endpoint

Answer a question across all collections (medline, plazi, pmc), ranked best-first.
`POST /qa/multi` is the same without the 120-second cache.

**Request:**
```json
{
  "question": "What causes malaria?",
  "mode": "generative",
  "depth": "fast"
}
```

| Field | Values | Default |
|-------|--------|---------|
| `mode` | `"extractive"` · `"generative"` | `"generative"` |
| `depth` | `"fast"` (two searches; about 3 s for one collection, 6 s for three) · `"deep"` (+ model-written searches; about 5 s, 8 s for three) | `"fast"` |
| `retrieval` | ignored (`"sparse"`, `"dense"` still accepted) | — |
| `retrieval_n` | documents per search before reranking | 20 |
| `final_n` | documents the reader gets | 5 |
| `doc_refs` | list of strings | `null` — triggers direct-document mode (see below) |

**Response:**
```json
{
  "question": "What causes malaria?",
  "mode_used": "generative",
  "ndocs_retrieved": 102,
  "model": "qwen3.8-27b",
  "pipeline_time": 3.1,
  "collection_results": [
    {
      "collection": "medline",
      "rank": 1,
      "mode_used": "generative",
      "depth_used": "fast",
      "notes": [],
      "queries_used": [],
      "answers": [
        {
          "answer": "Based on the provided documents, malaria is caused by Plasmodium parasites ...",
          "generation_confidence": 0.93,
          "hedged": false,
          "states_no_answer": false,
          "docs": [
            {
              "docid": "12345678",
              "doc_source": "medline",
              "doc_retrieval_score": 0.98,
              "doc_text": "...",
              "passage": "..."
            }
          ]
        }
      ]
    }
  ],
  "unresolved_refs": null
}
```

**Answer shape:**
- **Extractive**: `docs` has 1 element; `snippet_start`/`snippet_end` are char offsets into `doc_text`; `answer_score` is BioBERT confidence
- **Generative**: `docs` lists the documents the answer cites; `snippet_start`/`snippet_end` are `null`; `answer_score` is `null`

**What ran:** `depth_used` is the depth that actually ran, `mode_used` the reader that wrote the
answer. When the language model is slow or down, a generative request gets the extractive
answer at once: `mode_used` says `extractive` and `notes` holds
`{"code": "extractive_fallback", "message": "..."}`. When the deep mode's extra searches are
skipped for the same reason, `notes` holds `depth_reduced`. Show `message` as it is.

**Batch jobs** are recognised automatically: a POST that does not come from a web browser
(a script, such as the Kybele trait pipeline; browsers send `Sec-Fetch-Mode` and `Origin`).
Its answers are written at low priority on the shared server, after people's, and it may wait
up to about 100 s (`[llm] batch_*` in `config.toml`). When the server's gateway says it is full
or restarting (429 or 503, with `Retry-After`), a batch job waits as asked and sends the request
again, within that time. It never gets BioBERT's answer in place of a generated one: if the
model still does not answer, the response has no answers and says why in `error`. Requests from the web page keep the 4-second fallback.

### Direct-document mode (`doc_refs`)

Skip retrieval and target specific documents directly. Accepts PMID, PMCID, DOI, or a title fragment. Multiple refs from different collections are supported.

```json
{
  "question": "What is the body length of Zeuxevania hubeni?",
  "mode": "extractive",
  "doc_refs": ["36273439"]
}
```

```json
{
  "question": "What is the host of Plasmodium falciparum?",
  "mode": "generative",
  "doc_refs": ["PMC9712345", "12345678", "Zeuxevania hubeni new species"]
}
```

| Ref format | Collection searched | Match strategy |
|---|---|---|
| All digits (1–8 chars) | medline | Exact PMID match |
| `PMC\d+` | pmc | Exact PMCID match |
| Starts with `10.` | all | Exact DOI match |
| Anything else | all | Highest-scoring keyword hit |

Unresolvable refs are reported in `unresolved_refs` in the response; resolved docs still get answered.

QA runs on the document's full text (with IDF passage window), not just the abstract.

### `GET /api/QA`

One collection; backwards-compatible with `biodiversitypmc.sibils.org/api/QA`. The web
interface calls it once per collection tab. `n` is the number of documents read (at most 5);
`depth`, `depth_used`, `notes` and `queries_used` work as in `POST /qa`.

```
GET /api/QA?q=What+causes+malaria%3F&col=medline&n=5&mode=extractive&depth=fast
```

### `POST /batch`

Extractive QA over a list of questions with parallel retrieval.

```json
{
  "questions": ["What causes malaria?", "What diseases are associated with ticks?"],
  "final_n": 5
}
```

### `GET /health`

Service status, the commit the code was installed from (`commit`), and the models in use.

---

## Configuration

[`config.toml`](config.toml) — the key sections (every key has a default in `config.py`):

```toml
[generation]
backend = "remote"          # the shared server; "vllm" loads [model] in this process

[llm]
base_url = "http://llm.lan.text-analytics.ch/qwen/v1"
model = "Qwen3.8-27B"
first_token_timeout = 4.0   # then the extractive answer is sent, with a note

[depth]
default = "fast"
candidates_per_search = 20

[sibils]
collections = ["medline", "plazi", "pmc"]
cache_dir = "data/sibils_cache"

[reranking]
model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
device = "cpu"

[relevance_filter]
final_n = 5
```

Logs: one `qa_request` line per question with depth, mode, timings and note codes; never the
question text, which is also removed from the access log's request line.

---

## Project structure

```
BioMoQA-RAG/
├── src/biomoqa_rag/
│   ├── api_server.py           # FastAPI server + endpoints
│   ├── pipeline.py             # RAG pipeline orchestration
│   ├── depth.py                # fast / deep search: model-written queries, choosing the five
│   ├── llm_remote.py           # client for the shared language-model server
│   ├── config.py               # Config models (reads config.toml)
│   ├── passage.py              # passage selection shared by both readers
│   ├── answer_signals.py       # hedged / "not stated" signals, collection ranking
│   ├── retrieval/
│   │   ├── sibils_retriever.py # SIBILS API client: parser query, plain OR search (disk-cached)
│   │   ├── doc_resolver.py     # Resolve PMID/PMCID/DOI/title → Document
│   │   ├── reranker.py         # CrossEncoder reranker
│   │   └── query_parser.py     # SIBILS query parser integration
│   └── extraction/
│       └── extractive_qa.py    # BioBERT span extraction
├── tests/                      # offline tests (no request leaves the machine)
├── eval/                       # Evaluation scripts (run from repo root)
├── data/
│   └── sibils_cache/           # SIBILS API disk cache (7-day TTL)
├── results/                    # Evaluation outputs
├── config.toml                 # Runtime configuration
└── pyproject.toml              # Package + dependencies
```

Tests: `uv pip install pytest pytest-cov`, then `python -m pytest`.

---

## Deployment

Managed via Ansible/Jenkins (`bitem-heg-geneve/ansible-cfg`): branch `master` goes to dev,
`prod_5.1.5.10` to production. The service runs as `sibils-qa.service`.

The models are installed by ansible-cfg as pinned artifacts (`bitem.artifacts`, ansible-cfg
PR 143), linked as `/opt/sibils-qa/artifacts/<name>`, and `config.toml` there loads each one
from its folder: the service downloads nothing. With `generation.backend = "remote"` it loads
only the reranker and BioBERT, and the host must reach the shared language-model server.

The `config.toml` deployed to the server is the one in ansible-cfg
(`files/playbook_dev_sibils_qa/config.toml`); the one in this repository is for local runs and
lists every setting with its default.

---

## Troubleshooting

```bash
# Service status and logs
systemctl status sibils-qa
journalctl -u sibils-qa -f

# GPU usage
nvidia-smi
```
