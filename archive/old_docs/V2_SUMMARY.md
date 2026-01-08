# BioMoQA RAG V2 - Enhanced Pipeline Summary

## What We Built

**V2 is a significantly improved RAG pipeline** that addresses all 4 weaknesses identified in V1:

### ✅ Fixed Weaknesses

| Weakness (V1) | Solution (V2) | Implementation |
|---------------|---------------|----------------|
| **BM25 only** | Semantic reranking | Cross-encoder scores (query, doc) pairs |
| **No reranking** | Added reranker | `SemanticReranker` with ms-marco model |
| **No query expansion** | LLM + rule-based | Expands acronyms & generates synonyms |
| **No relevance filtering** | Keyword + LLM filtering | Filters irrelevant docs before generation |

---

## New Components

### 1. **Semantic Reranker** (`src/retrieval/reranker.py`)

**What it does:**
- Takes top-100 from SIBILS
- Scores each document against the question using cross-encoder
- Returns top-30 by semantic relevance (not just BM25 score)

**Models available:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, default)
- `cross-encoder/ms-marco-MedMarco-electra-base` (slower, better for biomedical)

**Expected improvement:** +10-15%

```python
reranker = SemanticReranker()
docs = sibils.retrieve(question, n=100)
docs = reranker.rerank(question, docs, top_k=30)
```

---

### 2. **Query Expander** (`src/retrieval/query_expander.py`)

**What it does:**
- Expands acronyms (AG1-IA → anastomosis group 1 IA)
- Generates synonym variants using LLM
- Retrieves with multiple query phrasings
- Deduplicates results

**Example:**
- Original: "What is AG1-IA?"
- Expanded:
  - "What is anastomosis group 1 IA?"
  - "What is the Rhizoctonia solani AG1-IA strain?"

**Expected improvement:** +15-20% (better recall)

```python
expander = HybridQueryExpander(llm)
expanded = expander.expand(question)  # Returns 2-3 queries
all_docs = []
for query in expanded.all_queries:
    all_docs.extend(sibils.retrieve(query, n=30))
```

---

### 3. **Relevance Filter** (`src/retrieval/relevance_filter.py`)

**What it does:**
- Filters out documents that don't help answer the question
- Three modes:
  - **Fast**: Keyword overlap (default)
  - **LLM**: Asks LLM "Is this relevant?"
  - **Hybrid**: Both (2-stage filtering)

**Expected improvement:** +10-20% (cleaner context)

```python
filter = FastRelevanceFilter(min_overlap=0.15)
relevant = filter.filter_relevant(question, docs, max_docs=20)
```

---

## V2 Pipeline Flow

```
Question: "What is AG1-IA?"
    ↓
[1] Query Expansion
    ├─ Original: "What is AG1-IA?"
    ├─ Acronym: "What is anastomosis group 1 IA?"
    └─ LLM variant: "What is the Rhizoctonia solani strain AG1-IA?"
    ↓
[2] Multi-Query Retrieval
    ├─ Query 1 → 35 docs
    ├─ Query 2 → 35 docs
    ├─ Query 3 → 35 docs
    └─ Deduplicate → 100 unique docs
    ↓
[3] Semantic Reranking
    ├─ Cross-encoder scores all 100 docs
    └─ Keep top-30 by relevance
    ↓
[4] Relevance Filtering
    ├─ Keyword overlap check
    └─ Keep top-20 relevant docs
    ↓
[5] Generation (vLLM)
    └─ Generate answer with citations
```

---

## Configuration

```python
config = RAGConfigV2(
    # Retrieval
    retrieval_n=100,          # Retrieve 100 initially

    # Query expansion
    use_query_expansion=True,
    n_query_variants=1,       # 1 LLM variant + 1 acronym expansion

    # Reranking
    use_reranking=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_n=30,              # Top-30 after reranking

    # Filtering
    use_relevance_filter=True,
    relevance_filter_type="fast",  # "fast", "llm", or "hybrid"
    final_n=20,               # Final docs for generation

    # Generation (same as V1)
    model_name="Qwen/Qwen2.5-7B-Instruct",
    use_vllm=True,
    gpu_memory_utilization=0.8,
)
```

---

## Expected Performance

### Speed
- **V1:** ~7s per question
- **V2:** ~8-12s per question (+1-5s slower)
- **Tradeoff:** Slightly slower but much better quality

### Quality (ROUGE-1 expected)
- **V1:** ~40%
- **V2:** ~60-70% (+40-60% improvement)

### Breakdown of Improvements
| Component | Gain | Cumulative |
|-----------|------|------------|
| Baseline (V1) | -- | 40% |
| + Reranking | +10-15% | 50-55% |
| + Query expansion | +15-20% | 65-75% |
| + Relevance filter | +10-20% | **75-95%** |

*Note: Gains are not strictly additive, may be less*

---

## How to Use

### Start V2 API Server

```bash
cd /home/egaillac/BioMoQA-RAG

# Kill old server
lsof -ti :9000 | xargs kill -9

# Start V2
./venv/bin/python3 -m uvicorn api_server_v2:app --host 0.0.0.0 --port 9000
```

### Test V2 vs V1

```bash
# Run comparison test on 5 questions
./venv/bin/python3 test_v2_improvements.py
```

### Use in Python

```python
from src.pipeline_vllm_v2 import EnhancedRAGPipeline, RAGConfigV2

pipeline = EnhancedRAGPipeline(RAGConfigV2())

result = pipeline.run(
    "What is AG1-IA?",
    debug=True  # Show retrieval steps
)

print(result["answer"])
print(f"Time: {result['pipeline_time']}s")
print(f"Debug: {result['debug_info']}")
```

---

## Files Created

```
BioMoQA-RAG/
├── src/
│   ├── v1_archive/
│   │   ├── pipeline_vllm_v1.py      # V1 backup
│   │   └── pipeline_v1.py           # V1 backup
│   │
│   ├── pipeline_vllm_v2.py          # V2 MAIN PIPELINE
│   │
│   └── retrieval/
│       ├── reranker.py              # NEW: Semantic reranking
│       ├── query_expander.py        # NEW: Query expansion
│       └── relevance_filter.py      # NEW: Relevance filtering
│
├── api_server.py                    # V1 API (original)
├── api_server_v2.py                 # V2 API (enhanced)
├── test_v2_improvements.py          # V1 vs V2 test script
│
└── docs/
    ├── RAG_EXPLAINED.md             # How RAG works
    ├── IMPROVEMENTS_ROADMAP.md      # Implementation guide
    └── V2_SUMMARY.md                # This file
```

---

## Next Steps

### 1. Test V2 (Do This First!)

```bash
# Quick test on 5 questions
./venv/bin/python3 test_v2_improvements.py
```

Expected output:
```
V1 vs V2 Comparison
--------------------------------------------------------------------------------
Question                                           V1 Time      V2 Time      Diff
--------------------------------------------------------------------------------
What is AG1-IA?                                     7.23s        9.45s      +2.2s
What causes corn sheath blight?                     6.89s       10.12s      +3.2s
...
AVERAGE                                             7.05s        9.80s      +2.8s
```

### 2. Full Evaluation (120 QA pairs)

Create `process_120_qa_v2.py`:
```python
# Similar to process_120_qa_via_api.py but using V2 pipeline
# Compare V1 vs V2 ROUGE scores
```

### 3. Deploy V2

If improvements confirmed:
```bash
# Update start script to use V2
./start_api.sh  # Point to api_server_v2.py

# Or run both (V1 on port 9000, V2 on port 9001)
```

---

## Troubleshooting

### GPU Memory Issues

V2 uses more GPU memory (reranker model + LLM):
```python
# Reduce memory usage
config = RAGConfigV2(
    gpu_memory_utilization=0.6,  # Lower from 0.8
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Use smaller model
)
```

### Too Slow

Disable expensive components:
```python
config = RAGConfigV2(
    use_query_expansion=False,     # Disable for +2s speedup
    relevance_filter_type="fast",   # Use fast (not LLM)
)
```

### LLM Filtering Too Aggressive

Adjust thresholds:
```python
filter = FastRelevanceFilter(min_overlap=0.10)  # Lower threshold
# Or
filter = LLMRelevanceFilter(min_relevant=10)  # Keep at least 10 docs
```

---

## Summary

**V2 Status:** ✅ Implemented, ready to test

**Key improvements:**
1. ✅ Semantic reranking (cross-encoder)
2. ✅ Query expansion (LLM + rules)
3. ✅ Relevance filtering (keyword-based)
4. ✅ Better document selection

**Expected gains:** +40-60% quality improvement

**Next action:** Run `test_v2_improvements.py` to verify

**Deployment:** After testing, switch API to V2 or run both versions

---

*V2 created: 2026-01-07*
*Author: ecsltae*
*Based on: Improvements Roadmap*
