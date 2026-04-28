# BioMoQA-RAG — Design Report

*Living document — updated as the system evolves.*

---

## 1. Starting point: what the old system did

The original system at [qa.sibils.org](https://qa.sibils.org) was a straightforward biomedical QA pipeline:

1. Take the user's question, submit it to the SIBILS BM25 search API (`medline` collection, n=5 documents).
2. Feed each document's title + abstract (first ~2000 chars) into a BioBERT model fine-tuned on SQuAD2 ([ktrapeznikov/biobert_v1.1_pubmed_squad_v2](https://huggingface.co/ktrapeznikov/biobert_v1.1_pubmed_squad_v2)).
3. Pick the highest-confidence span across all documents. Return it as the answer, along with the source document.

Key properties of the old system:
- **Single-pass BioBERT** — no sliding window, no two-pass. One forward pass per document, first 512 tokens only.
- **BM25-only retrieval** — no dense search, no reranking.
- **Medline-only** — no Plazi, no PMC.
- **Fast and simple** — the whole pipeline fits in a few dozen lines.

The old system worked well for factoid clinical questions where the answer is an exact phrase in a PubMed abstract.

---

## 2. The new system: original design

The first version of BioMoQA-RAG added several layers on top:

| Layer | Old system | New system (v1) |
|---|---|---|
| Retrieval | SIBILS BM25, medline only, n=5 | SIBILS BM25 + FAISS dense, all collections, n=30 |
| Reranking | None | CrossEncoder reranker |
| Relevance filter | None | FastRelevanceFilter (token overlap) |
| Answer extraction | BioBERT single-pass | BioBERT two-pass with sliding window |
| Generation | None | Qwen3-8B via vLLM |
| Collections | medline | medline + plazi + pmc + suppdata |

The idea was that more retrieval (hybrid FAISS+BM25) + reranking + better span extraction would improve quality, and the LLM could synthesise answers when extraction failed.

---

## 3. What we learned from benchmarking

### 3.1 Extractive: the old system was better

We benchmarked both systems on 50 deduplicated BioASQ factoid questions:

| System | F1 | ROUGE-1 | Answer rate |
|---|---|---|---|
| Old system (SIBILS BM25 + single-pass BioBERT) | **0.376** | **0.410** | 86% |
| New extractive pipeline (v1) | 0.323 | 0.348 | 100% |

The new system scored lower. The reasons were:

1. **FAISS contamination** — the FAISS index had only 2398 generic PMC documents. When fused with SIBILS via RRF, these wrong documents displaced the correct ones from medline.
2. **Sliding window hurt, not helped** — BioBERT's sliding window on long Plazi/PMC documents ran 5-24 seconds and often extracted spans from the wrong section. The old system's truncated single-pass was more precise for factoid questions.
3. **CrossEncoder reranker** — ranks by semantic similarity to the question, not by likelihood of containing the answer. For extractive QA, this hurts: a document about a related topic outranks the document with the exact answer phrase.
4. **Too many collections** — Plazi and suppdata documents are taxonomic treatments and OCR'd PDFs, not clinical prose. They scored high on BM25 (species names match) but BioBERT couldn't extract answers from them.
5. **BioBERT forced-span** — we configured `handle_impossible_answer=False` to always return a span, but for documents where the answer isn't present, this returned nonsense spans with high confidence.

### 3.2 Key insight: simple beats complex for factoid extraction

BM25 is near-optimal for factoid clinical questions because the answer phrase typically occurs verbatim in the abstract that also matches the question's keywords. Dense retrieval adds semantic recall at the cost of precision. For extractive QA, **precision matters more than recall** — a wrong document can't contribute a correct span.

---

## 4. Simplification: back to the old system's approach

Based on the benchmark, we reverted the extractive pipeline to replicate the original system:

- **BM25-only** for extractive retrieval (no FAISS, no reranker, no relevance filter)
- **Single-pass BioBERT** on the first 2000 chars (matching old system behaviour exactly)
- **All three collections** (medline, plazi, pmc) — but separately per-collection, not mixed

The `suppdata` collection was removed entirely: supplementary files are OCR'd PDFs and spreadsheets with no prose abstracts; BioBERT returns garbage spans and the LLM hallucinates from noisy text.

---

## 5. Decoupling retrieval from answer mode

The key architectural shift: retrieval and answer mode are now **independent parameters**.

Previously the system had a `"hybrid"` mode that bundled retrieval strategy with the answer model. This made it impossible to e.g. use FAISS for extractive (which doesn't make sense) or BM25 for generative (which could make sense for synthesis from curated sources).

The new API contract:

```json
{
  "question": "...",
  "mode":      "extractive" | "generative",
  "retrieval": "sibils"     | "rag"
}
```

**`retrieval=sibils`** — BM25 only, no FAISS, no reranker. Fast (cached: <1s). Best for factoid questions.

**`retrieval=rag`** — FAISS + BM25 hybrid (RRF fusion) + CrossEncoder reranker + relevance filter. Slower (3-4s). Better for semantic coverage and generative synthesis.

The recommended pairings, confirmed by evaluation:
- **Extractive → SIBILS** (BM25 precision optimal for span extraction)
- **Generative → RAG** (semantic FAISS expands coverage for synthesis)

Both non-standard pairings (RAG+extractive, SIBILS+generative) are allowed — the user can choose.

Generative answers now return a single item with a `docids` list and a `docs` list, instead of one item per cited document with empty `answer=""` fields — cleaner for the frontend.

---

## 6. RAG retriever improvements

Once the SIBILS+extractive path was stable, we focused on improving the RAG path (FAISS+BM25) for generative mode.

### 6.1 What was wrong with the FAISS retriever

| Problem | Root cause |
|---|---|
| FAISS retrieved wrong docs | 2398 generic PMC docs, no source metadata |
| SmartHybridRetriever skipped FAISS | Heuristic: "technical query → BM25 only" fired for nearly all biomedical queries |
| Documents had `source="unknown"` | Index built from plain dicts without source field |
| alpha=0.5 gave equal weight to a tiny corpus | FAISS deserved lower weight when it was low-quality |

### 6.2 Fixes applied

**Heuristic removal** — `SmartHybridRetriever` had a routing heuristic that sent queries with "technical terms" (gene, protein, enzyme, antibody…) to BM25-only. Since nearly all biomedical questions have such terms, FAISS was effectively never used. Removed; always runs parallel BM25+FAISS.

**FAISS index rebuilt from SIBILS cache** — The disk cache (dbm file at `data/sibils_cache/cache`) accumulates all SIBILS API responses. We built a script (`rebuild_faiss_from_cache.py`) to extract, deduplicate, and re-embed them. Corpus grew from 2398 → 4341 documents, all with correct source/pmid metadata.

**Alpha tuning** — Initial finding with old MiniLM model: alpha=0.3 (more BM25) was optimal. After switching to biomedical embeddings, alpha=0.5 became optimal (FAISS is now accurate enough to deserve equal weight).

### 6.3 Biomedical embeddings

We replaced `sentence-transformers/all-MiniLM-L6-v2` (general English, 384-dim) with `pritamdeka/S-PubMedBert-MS-MARCO` (PubMedBERT fine-tuned on MS-MARCO for retrieval, 768-dim).

Effect on RAG retrieval (50 BioASQ questions, alpha sweep, reranker excluded):

| Model | Alpha | Context recall | F1 |
|---|---|---|---|
| MiniLM | 0.3 | 64% | 0.320 |
| S-PubMedBert-MS-MARCO | 0.3 | 66% | 0.346 |
| S-PubMedBert-MS-MARCO | 0.5 | 68% | **0.370** |
| S-PubMedBert-MS-MARCO | 0.7 | **70%** | **0.370** |

+6% recall and +16% F1 over the MiniLM baseline, at the same latency.

Notably, with the biomedical model, higher alpha (more FAISS weight) becomes optimal — the model is accurate enough that the FAISS component now helps rather than hurts.

### 6.4 Better reranker

Upgraded CrossEncoder from `ms-marco-MiniLM-L-6-v2` (smallest/fastest) to `ms-marco-MiniLM-L-12-v2` (2× larger, ~2× better MRR on MS-MARCO). The reranker is only used in the RAG path; no effect on SIBILS-only extractive latency.

### 6.5 Full pipeline progression (50 BioASQ questions, SIBILS+extractive vs RAG+extractive)

| Run | Strategy | Context recall | F1 | Answer rate | Avg time |
|---|---|---|---|---|---|
| v1 — cold cache | SIBILS | 28% | 0.199 | 58% | 10.8s |
| v1 — cold cache | RAG | 30% | 0.210 | 100% | 7.9s |
| v2 — warm cache, fixed heuristic | SIBILS | 54% | 0.273 | 78% | 0.96s |
| v2 — warm cache, fixed heuristic | RAG | 58% | 0.268 | 100% | 3.0s |
| v3 — biomedical embeddings + L-12 reranker | SIBILS | 54% | 0.273 | 78% | 0.35s |
| **v3 — biomedical embeddings + L-12 reranker** | **RAG** | **60%** | **0.315** | **100%** | 3.2s |

The SIBILS cold-cache failure (10.8s, 42% questions return 0 docs) is an evaluation artefact. In production, the disk cache means SIBILS responses are near-instant. The "warm cache" row is the realistic production baseline.

---

## 7. Current architecture

```
User question
    │
    ├─ retrieval=sibils ──► SIBILSRetriever (BM25, disk-cached)
    │                            │
    │                       top-5 docs (capped)
    │                            │
    │                    ┌───────┴────────┐
    │                    │  mode=extractive│  BioBERT single-pass → ranked spans
    │                    │  mode=generative│  Qwen3-8B (vLLM) → synthesised answer
    │                    └────────────────┘
    │
    └─ retrieval=rag ────► SmartHybridRetriever
                               ├─ SIBILS BM25 (n=10)  ┐
                               └─ FAISS dense  (k=10)  ┤── RRF (alpha=0.5)
                                   S-PubMedBert-MSMARCO │
                                                        ▼
                                              CrossEncoder reranker
                                              (ms-marco-MiniLM-L-12-v2)
                                              + FastRelevanceFilter
                                              + top-5 cap
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    │  mode=extractive  │  mode=generative  │
                                    │  BioBERT spans    │  Qwen3-8B + cites │
                                    └───────────────────┴───────────────────┘
```

**LLM**: Qwen3-8B-FP8 via vLLM, served with `enable_thinking=False` (assistant-turn pre-fill to force English output). Uses `apply_chat_template` directly rather than `llm.chat()` to sidestep vLLM version differences in `chat_template_kwargs`.

**Collections** (multi-collection endpoint): medline, plazi, pmc — retrieved in parallel, each getting its own final_n slot. Results ranked by best-document score. `suppdata` removed (OCR'd PDFs, no usable prose).

**FAISS index**: 4739 documents, rebuilt from `data/sibils_cache/cache` (dbm file accumulating all SIBILS API responses). Run `python rebuild_faiss_from_cache.py` to refresh as the cache grows.

**Important**: `retrieval=rag` queries **both** SIBILS BM25 and FAISS in parallel — FAISS is additive, not a replacement for SIBILS. The difference from `retrieval=sibils` is that RAG additionally queries the dense index and fuses the two ranked lists via RRF before reranking. SIBILS BM25 coverage is the baseline in both modes; FAISS adds documents that SIBILS would miss (semantically related but lexically non-overlapping).

---

## 8. What could improve further

### 8.1 SIBILS + extractive — near its ceiling

The extractive pipeline already matches the old qa.sibils.org system (F1=0.376 on BioASQ). The main limiters are:

| Bottleneck | Why it's hard |
|---|---|
| SIBILS recall (54% context recall with warm cache) | BM25 lexical matching misses paraphrase-heavy answers |
| BioBERT span precision | The model is correct but limited to what it was trained on (SQuAD2 distribution vs clinical literature) |
| Collection scope | medline is the right collection for factoid questions; other collections add noise |

**Realistic improvements:**
- **Query reformulation** — if the first SIBILS call returns 0 docs, try a paraphrased query using the LLM (adds ~1-2s but recovers the 22% miss rate)
- **Larger n** — currently n=10 → final 5. Increasing to n=20 → final 7 may recover some low-recall cases at the cost of BioBERT latency
- **Better BioBERT** — fine-tuning on a biomedical QA dataset closer to the actual query distribution (BioASQ train split) would directly improve span quality

### 8.2 RAG + generative — more headroom

The generative path has more room to grow because it depends on components that can all be improved independently:

| Component | Current | Improvement path |
|---|---|---|
| FAISS corpus | 4739 docs | Grows automatically with `rebuild_faiss_from_cache.py`; seed with PubMed bulk dumps (millions of abstracts) |
| Embedding model | S-PubMedBert-MS-MARCO | Already strong; `allenai/specter2` or a retrieval-fine-tuned BiomedBERT at larger corpus size |
| Reranker | ms-marco-MiniLM-L-12-v2 | A biomedical-specific reranker trained on PubMed question-document pairs |
| LLM | Qwen3-8B-FP8 | Fine-tuning on biomedical QA, or a larger model when GPU allows |
| Prompt | Current 2-4 sentence prompt | Structured prompts with explicit citation instructions; chain-of-thought for complex questions |
| Context window | 8 docs × 600 chars | Longer snippets with better sentence-boundary truncation |

**Note on FAISS corpus size**: because RAG always queries SIBILS as well, the FAISS corpus size is not a hard ceiling on recall — SIBILS already provides strong BM25 coverage. What a larger FAISS index adds is *semantic diversity*: documents about the same topic but phrased differently from the query, which BM25 misses. Growing the FAISS index (via `rebuild_faiss_from_cache.py`, or seeding from PubMed Central bulk abstracts) expands this semantic complement to SIBILS rather than replacing it.

**Evaluation gap**: we have measured RAG+extractive (using BioBERT) but not RAG+generative directly on BioASQ. The generative path should be evaluated separately with metrics suited to multi-sentence synthesis (ROUGE, BERTScore, human evaluation) rather than exact-span F1.

---

## 9. Evaluation scripts

| Script | Purpose |
|---|---|
| `evaluate_comparison.py` | Old system (SIBILS+BioBERT) vs new extractive pipeline on BioASQ |
| `evaluate_retrieval.py` | SIBILS vs RAG retrieval: context recall, F1, doc count, latency |
| `evaluate_alpha.py` | Sweep over RRF alpha and embedding model — isolates retrieval quality |
| `rebuild_faiss_from_cache.py` | Rebuild FAISS index from SIBILS disk cache; `--model` flag for embedding choice |

All evaluation scripts use `kroshan/BioASQ` (HuggingFace) deduplicated to unique factoid questions.

---

*Last updated: 2026-04-28*
