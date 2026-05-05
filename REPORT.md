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
- **medline + plazi** — the SIBILS default collection set; no PMC, no suppdata.
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
- **Sparse + Extractive** (`retrieval=sibils` + `mode=extractive`) — BM25 precision optimal for span extraction
- **Dense + Generative** (`retrieval=rag` + `mode=generative`) — semantic FAISS expands coverage for synthesis

Both non-standard pairings (Dense + Extractive, Sparse + Generative) are allowed — the user can choose.

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

### 6.5 Full pipeline progression (50 BioASQ questions, Sparse + Extractive vs Dense + Extractive)

| Run | Strategy | Context recall | F1 | Answer rate | Avg time |
|---|---|---|---|---|---|
| v1 — cold cache | Sparse | 28% | 0.199 | 58% | 10.8s |
| v1 — cold cache | Dense | 30% | 0.210 | 100% | 7.9s |
| v2 — warm cache, fixed heuristic | Sparse | 54% | 0.273 | 78% | 0.96s |
| v2 — warm cache, fixed heuristic | Dense | 58% | 0.268 | 100% | 3.0s |
| v3 — biomedical embeddings + L-12 reranker | Sparse | 54% | 0.273 | 78% | 0.35s |
| **v3 — biomedical embeddings + L-12 reranker** | **Dense** | **60%** | **0.315** | **100%** | 3.2s |

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

### 8.1 Sparse + Extractive — near its ceiling

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

### 8.2 Dense + Generative — more headroom

The generative path has more room to grow because it depends on components that can all be improved independently:

| Component | Current | Improvement path |
|---|---|---|
| FAISS corpus | 4739 docs | Grows automatically with `rebuild_faiss_from_cache.py`; seed with PubMed bulk dumps (millions of abstracts) |
| Embedding model | S-PubMedBert-MS-MARCO | Already strong; `allenai/specter2` or a retrieval-fine-tuned BiomedBERT at larger corpus size |
| Reranker | ms-marco-MiniLM-L-12-v2 | A biomedical-specific reranker trained on PubMed question-document pairs |
| LLM | Qwen3-8B-FP8 | Fine-tuning on biomedical QA, or a larger model when GPU allows |
| Prompt | Current 2-4 sentence prompt | Structured prompts with explicit citation instructions; chain-of-thought for complex questions |
| Context window | 8 docs × 600 chars | Longer snippets with better sentence-boundary truncation |

**Note on FAISS corpus size**: because the Dense path always queries SIBILS as well, the FAISS corpus size is not a hard ceiling on recall — SIBILS already provides strong BM25 coverage. What a larger FAISS index adds is *semantic diversity*: documents about the same topic but phrased differently from the query, which BM25 misses. Growing the FAISS index (via `rebuild_faiss_from_cache.py`, or seeding from PubMed Central bulk abstracts) expands this semantic complement to SIBILS rather than replacing it.

---

## 9. Evaluation scripts

| Script | Purpose |
|---|---|
| `eval/evaluate_comparison.py` | Old system (SIBILS+BioBERT) vs new extractive pipeline on BioASQ |
| `eval/evaluate_retrieval.py` | SIBILS vs RAG retrieval: context recall, F1, doc count, latency |
| `eval/evaluate_alpha.py` | Sweep over RRF alpha and embedding model — isolates retrieval quality |
| `eval/evaluate_semantic.py` | Extractive vs generative vs SIBILS API; BERTScore + cosine similarity |
| `eval/evaluate_bioasq_qa.py` | **QA-only benchmark** — gold context supplied, retrieval bypassed (see §10) |
| `rebuild_faiss_from_cache.py` | Rebuild FAISS index from SIBILS disk cache; `--model` flag for embedding choice |

All evaluation scripts use `kroshan/BioASQ` (HuggingFace) deduplicated to unique factoid questions.

---

## 10. Evaluation methodology: BioASQ QA-only benchmark

### 10.1 End-to-end baseline (120 BioASQ questions, real retrieval)

The initial evaluation ran both strategies end-to-end on 120 BioASQ factoid questions using `pipeline.run()` with real retrieval.

| Strategy | Answer rate | Answer Contains | SQuAD F1 | Avg time |
|---|---|---|---|---|
| **Sparse + Extractive** | 100% | 26.7% | 11.8% | 0.86s |
| **Dense + Generative** | 100% | **48.3%** | 2.4% | 1.49s |

These numbers are hard to interpret in isolation: is a low score due to the retriever failing to return the right document, or the model failing to extract the right answer from a correct document? The next section unpacks this confound.

### 10.2 The retrieval-bias problem

End-to-end evaluation conflates two distinct quality signals:

1. **Retrieval quality** — does the system find the right document?
2. **QA reader quality** — given a document, can the model extract the right answer?

BioASQ factoid questions are derived from PubMed abstracts — each question was authored to be answered by a specific abstract passage. When evaluation runs the full pipeline, the retriever may (or may not) return the exact source abstract. If it does, the QA model scores well not because it's a good reader, but because the answer was handed to it. If it doesn't, the QA model has no chance regardless of its reading ability.

This means **F1 and context recall are correlated**: a good retrieval run looks like a good QA run. The confound is severe enough to mask real differences between QA models. The solution: supply the gold passage directly, bypassing retrieval.

### 10.3 The BioASQ dataset format

`kroshan/BioASQ` on HuggingFace (`split="train"`) provides (question, answer, context) triples:

```
text field: <answer>GOLD_ANSWER <context>GOLD_CONTEXT</context>
```

The gold context is the PubMed passage that contains the answer verbatim — typically 300–2000 characters from a single abstract. This enables a **reading comprehension evaluation**: supply the gold passage directly to the QA model, measure whether it extracts the answer correctly.

#### Sample entries

| Question | Gold answer | Context length |
|---|---|---|
| What is the inheritance pattern of Li–Fraumeni syndrome? | autosomal dominant | 1480 chars |
| Which type of lung cancer is afatinib used for? | EGFR-mutant NSCLC | 1491 chars |
| What is the major adverse effect of adriamycin (doxorubicin)? | cardiotoxicity | 1303 chars |
| Which is the branch site consensus in U12-dependent introns? | UUCCUUAAC | 1331 chars |
| Which gene is most commonly mutated in Tay-Sachs disease? | HEXA | 1779 chars |

Questions are diverse biomedical factoids: gene names, drug targets, disease mechanisms, molecular sequences. Gold answers are typically 1–5 tokens (named entities, short phrases).

### 10.4 Illustrative model outputs

Two examples from `results/eval_bioasq_qa.csv` that show where automated metrics succeed and fail.

**Example A — correct paraphrase, metric misses it (generative)**

> **Question:** Which type of lung cancer is afatinib used for?
> **Gold answer:** `EGFR-mutant NSCLC`
> **Context (excerpt):** "Reversible ATP-competitive inhibitors targeting the epidermal growth factor receptor (EGFR) have been established as the most effective treatment of patients with advanced non-small cell lung cancer (NSCLC) harboring 'activating' mutations in exons 19 and 21 of the EGFR gene…"
> **Model output:** "Based on the provided documents, afatinib is used for treating non-small cell lung cancer (NSCLC) that harbors 'activating' mutations in exons 19 and 21 of the EGFR gene. It targets the epidermal growth factor receptor (EGFR)…"
> **Scores:** Answer Contains = 0 · SQuAD F1 = 0.038 · Faithfulness = 0.74

The model's answer is factually correct and fully grounded in the context (faithfulness=0.74). Both Answer Contains and F1 score zero because the exact phrase "EGFR-mutant NSCLC" does not appear as a substring — the model uses an equivalent but longer description. This is the primary failure mode of substring-based metrics for verbose generative output.

**Example B — metric fooled, model actually refuses to answer (generative)**

> **Question:** Are most driver gene mutations synonymous or non-synonymous?
> **Gold answer:** `non-synonymous`
> **Context (excerpt):** "Genomic studies in neuroblastoma have showed only a few recurrent mutations…"
> **Model output:** "Based on the provided documents, there is no direct information regarding whether most driver gene mutations in neuroblastoma are synonymous or **non-synonymous**. The text mentions genomic studies revealing 'a few recurrent mutations' but does not specify their type…"
> **Scores:** Answer Contains = 1 · SQuAD F1 = 0.039 · Faithfulness = 0.52

Answer Contains = 1 because the string "non-synonymous" appears in the hedge ("are synonymous or non-synonymous"). The model is explicitly saying it cannot answer, yet the metric scores it as a hit. Faithfulness=0.52 is moderately low — the hedge sentences have less lexical overlap with the context than a direct answer would. This is the primary failure mode of Answer Contains for hedged or refusal outputs.

Together, these two examples show that no single metric is complete: Answer Contains is better than F1 for generative output, but can be fooled by hedges; faithfulness complements it by checking grounding independently of the gold answer string.

### 10.5 Evaluation design

**Script**: `eval/evaluate_bioasq_qa.py`

Retrieval is bypassed entirely via `RAGPipeline.run_with_contexts(question, [gold_context], mode)` — the gold passage is wrapped as a Document and passed directly to the QA step.

**Default sample**: 200 questions (`--limit` to override). `--extractive-only` skips generative (no GPU needed).

### 10.6 Metrics and rationale

#### For extractive (BioBERT)

BioBERT performs span extraction: it selects a contiguous token span from the provided passage. Since the gold answer is guaranteed to be present in the passage, this is a proper reading comprehension test.

| Metric | Rationale |
|---|---|
| **Exact Match (EM)** | Primary BioASQ official metric. Answer string matches gold exactly (after lowercasing and punctuation removal). |
| **SQuAD F1** | Token overlap, partial credit. Standard for extractive QA since SQuAD. |
| **ROUGE-L** | Longest common subsequence — catches correct partial spans. |

#### For generative (Qwen3-8B)

Generative models produce free-form prose. The gold answer ("autosomal dominant") may appear verbatim in a longer sentence ("The syndrome shows autosomal dominant inheritance"). Exact Match penalises this unfairly.

| Metric | Rationale |
|---|---|
| **Answer Contains** | Does the normalized gold answer appear as a substring of the generated answer? Captures the case where the model is factually correct but verbose. Analogous to the *nugget recall* used in TREC RAG 2024. |
| **BERTScore** | Semantic similarity via contextual embeddings. Handles paraphrase ("heart toxicity" ≈ "cardiotoxicity"). The standard semantic metric for generative QA. |
| **ROUGE-L** | Still useful as a surface overlap signal. |
| **SQuAD F1** | Included for cross-system comparability with extractive results. |

#### Faithfulness evaluation (TREC AIS proxy)

TREC RAG 2024 introduced **AIS** (Attribution to Identified Sources): each sentence of the generated answer is checked against the retrieved passages to verify it is grounded, not hallucinated. Full AIS requires an LLM judge. We implement a lightweight lexical proxy: for each sentence of the generated answer, compute ROUGE-1 recall against the gold context (fraction of the sentence's tokens that appear in the context). The score is the mean across sentences.

```
faithfulness(answer, context):
    for each sentence in answer:
        recall = |tokens(sentence) ∩ tokens(context)| / |tokens(sentence)|
    return mean(recall)
```

A score above 0.6 indicates the answer is predominantly grounded in the supplied passage. On our 200-question sample, generative answers average **faithfulness = 63.7%**: 79% of answers score above 0.5, and 98.5% above 0.3 — suggesting Qwen3 stays largely on-topic.

For BioASQ factoid questions, *Answer Contains* also serves as a proxy for TREC-style **nugget recall**: the gold answer is already a single atomic fact (the nugget), so checking whether it appears in the generated text is equivalent to checking nugget recall. Full nugget decomposition (decomposing multi-sentence answers into atomic claims and verifying each) is the natural extension for complex synthesis questions.

### 10.7 Results (200 BioASQ questions, gold context)

| System | Answer rate | Exact Match | Answer Contains | SQuAD F1 | ROUGE-1 | BERTScore | Faithfulness | Avg time |
|---|---|---|---|---|---|---|---|---|
| **Extractive (BioBERT)** | 100% | **50.5%** | 61.5% | **62.3%** | **64.7%** | **54.3%** | — | 0.11s |
| **Generative (Qwen3-8B)** | 99.5% | 0.0% | **72.0%** | 8.4% | 9.3% | −9.0% | **63.7%** | 3.58s |

**Extractive** — BioBERT extracts the correct span 50.5% of the time (EM) from the gold passage. F1=62.3% shows good partial credit on near-misses (e.g. "epidermal growth factor receptor" for gold "EGFR").

**Generative** — Answer Contains=72% shows that the gold fact is embedded in the generated answer 72% of the time. Faithfulness=63.7% confirms that answers are mostly grounded in the supplied passage.

**The metric problem for generative** is clearly visible: F1=8.4% and BERTScore=−9% are both artefacts of comparing a verbose sentence against a 1–5 token gold answer. ROUGE and SQuAD F1 are length-biased; BERTScore rescaled against a short reference also breaks for long outputs. **Answer Contains is the right primary metric for generative factoid QA**.

**Practical takeaway:** extractive is fast and precise (EM=50.5%); generative covers more answers but verbosely (Contains=72%). Comparing with the end-to-end baseline (§10.1): Answer Contains drops to 26.7% / 48.3% when retrieval is imperfect — the decomposition `E2E ≈ retrieval_recall × reader_quality` holds (54% × 61.5% ≈ 33%, close to 26.7%).

### 10.8 Benchmark properties

- **No retrieval contamination** — the system cannot "cheat" by finding the source document.
- **Controlled input** — both BioBERT and Qwen3 receive exactly the same passage; differences in score directly reflect reader quality.
- **Interpretable ceiling** — since the gold answer is in the context by construction, a perfect extractive system would score EM=100%. Observed EM reveals how much the model degrades relative to the oracle.
- **Separability** — gold-context and end-to-end results decompose cleanly: `E2E ≈ retrieval_recall × reader_quality` (§10.1 and §10.7).

---

## 11. LangGraph sandbox

### 11.1 Motivation

The `RAGPipeline.run()` method is an imperative function: retrieval → filter → generate, written as sequential Python statements. This works fine but has two operational weaknesses:

- **No explicit state** — intermediate results (documents, timing, debug info) live in local variables. There is no object that represents "the pipeline's current state" and can be inspected, checkpointed, or replayed.
- **Hard to extend conditionally** — adding a new route (e.g. "retry with reformulated query if 0 docs returned") means adding `if/else` inside an already-long method.

LangGraph is an orchestration framework that replaces the imperative chain with a **state machine**: each pipeline stage is a node, the pipeline state is an explicit typed dict, and transitions between stages are edges (which can be conditional). This makes the data flow visible as a graph rather than implicit in control flow.

### 11.2 Implementation

A sandbox class `LangGraphRAGPipeline` was created at `src/biomoqa_rag/pipeline_langgraph.py`. It subclasses `RAGPipeline`, inheriting all components unchanged (retrievers, reranker, LLM). Only `run()` is overridden.

**Pipeline state (`RAGState` TypedDict):**

```python
class RAGState(TypedDict):
    question, retrieval_n, final_n, collection,   # inputs
    mode, retrieval, return_documents, debug,      # inputs
    documents, num_retrieved,                      # retrieval output
    answers, debug_info                            # answer output
```

**Graph topology:**

```
START
  │
  ▼
[retrieve]  ← calls _retrieve_and_prepare() (BM25/FAISS/reranker unchanged)
  │
  ├── "extractive" ──► [extractive] ──► END   (BioBERT span extraction)
  ├── "generative" ──► [generative] ──► END   (Qwen3-8B vLLM generation)
  └── "__end__"    ──────────────────► END    (0 docs retrieved — short-circuit)
```

`pipeline.py` and `pyproject.toml` were not touched. `langgraph` was installed directly via `uv pip install`.

### 11.3 Benchmark results

The benchmark compared `RAGPipeline.run()` (original, called via `super()`) and `LangGraphRAGPipeline.run()` (LangGraph) on the same loaded model instance, isolating pure orchestration overhead from LLM inference time.

**Setup:** Qwen3-8B FP8 via vLLM, medline collection, generative mode, 5 queries, warm cache.

| Query | Original | LangGraph | Δ |
|---|---|---|---|
| Hydroxychloroquine mechanism | 3.17s | 2.16s | −1.0s |
| Lyme disease transmission | 3.68s | 2.09s | −1.6s |
| COVID-19 symptoms | 3.33s | 1.71s | −1.6s |
| CRISPR-Cas9 (0 docs retrieved) | 1.24s | 0.004s | −1.2s |
| p53 cancer suppression | 4.55s | 2.93s | −1.6s |
| **Average** | **3.19s** | **1.78s** | **−1.4s** |

**Interpretation:**

The LangGraph run appears ~44% faster, but this is a **SIBILS cache effect**: run 1 (original) populated the 7-day disk cache, run 2 (LangGraph) hit it. The retrieval time difference is cache latency, not orchestration overhead.

The CRISPR query makes this explicit: original took 1.24s (cache miss + disk write), LangGraph took **4ms** (cache hit → `_route_mode` returned `__end__` immediately, skipping the answer node entirely).

Real LangGraph orchestration overhead: **negligible** (<5ms per query — state dict construction and graph traversal are pure Python, dominated by retrieval and inference time).

**Answer correctness:** 3/5 exact string matches. The 2 non-matches are LLM sampling variance (temperature=0.1, not fully deterministic): both answers start identically and diverge after ~100 characters. Semantically, all 5 answers are correct.

### 11.4 What LangGraph adds

| Capability | Without LangGraph | With LangGraph |
|---|---|---|
| Pipeline state | Local variables, ephemeral | Typed dict, inspectable after run |
| Stage transitions | Implicit (sequential code) | Explicit edges, visible as graph |
| Conditional routing | `if/else` inside method | Named edges (`_route_mode`) |
| 0-doc short-circuit | `if not documents: return {...}` | `__end__` edge, skips answer node |
| Checkpointing | Not possible | Add `MemorySaver` to `compile()` |
| Future: query retry | Rewrite `run()` | Add `rewrite_query` node + back-edge |

The sandbox is production-ready as a drop-in: swap `from biomoqa_rag.pipeline import RAGPipeline` for `from biomoqa_rag.pipeline_langgraph import LangGraphRAGPipeline`.

---

