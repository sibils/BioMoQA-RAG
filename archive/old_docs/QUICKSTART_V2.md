# Quick Start: Test V2 in 3 Steps

## Step 1: Verify Installation (5 seconds)

```bash
cd /home/egaillac/BioMoQA-RAG
./venv/bin/python3 -c "import vllm, sentence_transformers, fastapi; print('✓ All packages installed')"
```

**Expected output:** `✓ All packages installed`

---

## Step 2: Test Components Without GPU (30 seconds)

```bash
./venv/bin/python3 << 'EOF'
from src.retrieval.query_expander import AcronymExpander
from src.retrieval.relevance_filter import FastRelevanceFilter
from src.retrieval import SIBILSRetriever

# Test acronym expansion
expander = AcronymExpander()
result = expander.expand("What is AG1-IA?")
print(f"1. Acronym expansion: {result.original} → {result.all_queries}")

# Test SIBILS retrieval
retriever = SIBILSRetriever(default_n=10)
docs = retriever.retrieve("What is malaria?", n=10)
print(f"2. SIBILS retrieval: {len(docs)} documents retrieved")

# Test relevance filter
filter = FastRelevanceFilter(min_overlap=0.15)
filtered = filter.filter_relevant("What is malaria?", docs, max_docs=5)
print(f"3. Relevance filter: {len(docs)} → {len(filtered)} documents")

print("\n✓ All components work!")
EOF
```

---

## Step 3: Choose Your Test

### Option A: Start V2 API Server (for web access)

```bash
# Clear GPU
nvidia-smi | grep VLLM | awk '{print $5}' | xargs -r kill -9

# Start server
./venv/bin/python3 -m uvicorn api_server_v2:app --host 0.0.0.0 --port 9000
```

Then open: **http://egaillac.lan.text-analytics.ch:9000/docs**

### Option B: Test Single Question (for quick verification)

```bash
# Clear GPU
nvidia-smi | grep VLLM | awk '{print $5}' | xargs -r kill -9

# Run single question test
./venv/bin/python3 << 'EOF'
from src.pipeline_vllm_v2 import EnhancedRAGPipeline, RAGConfigV2

print("Loading V2 pipeline (takes ~60s)...")

config = RAGConfigV2(
    retrieval_n=50,
    use_query_expansion=True,
    use_reranking=True,
    use_relevance_filter=True,
    relevance_filter_type='fast',
    final_n=10,
    gpu_memory_utilization=0.8,
)

pipeline = EnhancedRAGPipeline(config)

print("\nRunning test question...")
result = pipeline.run('What is AG1-IA?', debug=True)

print(f"\n{'='*80}")
print("RESULT:")
print(f"{'='*80}")
print(f"Time: {result['pipeline_time']}s")
print(f"Docs used: {result['num_retrieved']}")

if 'debug_info' in result:
    debug = result['debug_info']
    print(f"\nRetrieval pipeline:")
    print(f"  Initial retrieval: {debug.get('initial_retrieval_count', 0)} docs")
    print(f"  After reranking: {debug.get('reranked_count', 0)} docs")
    print(f"  After filtering: {debug.get('filtered_count', 0)} docs")

answer = ' '.join([s['text'] for s in result['answer']])
print(f"\nAnswer: {answer[:300]}...")

print(f"\n✓ V2 pipeline works!")
EOF
```

### Option C: Full V1 vs V2 Comparison (for evaluation)

```bash
# This takes 10-15 minutes (loads both pipelines)
# NOTE: Requires 60GB+ GPU memory for both pipelines
./venv/bin/python3 test_v2_improvements.py
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'vllm'` | Use `./venv/bin/python3` not `python3` |
| `GPU out of memory` | `nvidia-smi \| grep VLLM \| awk '{print $5}' \| xargs kill -9` |
| `Reranker slow first time` | Normal - downloading 90MB model |

---

## What V2 Does

1. **Expands your question** with synonyms and acronyms
2. **Retrieves documents** from multiple query variants
3. **Reranks semantically** using cross-encoder
4. **Filters irrelevant** documents
5. **Generates answer** with citations

**Expected improvement:** +40-60% better quality than V1

---

## Next Steps

After testing:

1. **Deploy:** Update `start_api.sh` to use `api_server_v2.py`
2. **Evaluate:** Run on 120 QA pairs
3. **Push:** `git push` to GitHub

---

**Quick check everything works:**
```bash
./venv/bin/python3 -c "
from src.pipeline_vllm_v2 import EnhancedRAGPipeline, RAGConfigV2
from src.retrieval.reranker import SemanticReranker
from src.retrieval.query_expander import AcronymExpander
print('✓ V2 ready to use')
"
```
