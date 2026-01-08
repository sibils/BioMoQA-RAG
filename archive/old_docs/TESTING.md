# Testing BioMoQA RAG V2

## Quick Component Test

Test that all V2 components work without loading the full pipeline:

```bash
cd /home/egaillac/BioMoQA-RAG

# Test imports
./venv/bin/python3 -c "
from src.retrieval.reranker import SemanticReranker
from src.retrieval.query_expander import AcronymExpander
from src.retrieval.relevance_filter import FastRelevanceFilter
from src.pipeline_vllm_v2 import EnhancedRAGPipeline, RAGConfigV2
print('✓ All V2 components ready')
"
```

## Test Individual Components

### 1. Test Acronym Expansion

```bash
./venv/bin/python3 << 'EOF'
from src.retrieval.query_expander import AcronymExpander

expander = AcronymExpander()
result = expander.expand("What is AG1-IA?")

print(f"Original: {result.original}")
print(f"Expanded: {result.expansions}")
EOF
```

### 2. Test SIBILS Retrieval

```bash
./venv/bin/python3 << 'EOF'
from src.retrieval import SIBILSRetriever

retriever = SIBILSRetriever(default_n=10)
docs = retriever.retrieve("What is malaria?", n=10)

print(f"Retrieved {len(docs)} documents")
if docs:
    print(f"First: {docs[0].title}")
EOF
```

### 3. Test Reranker

```bash
./venv/bin/python3 << 'EOF'
from src.retrieval import SIBILSRetriever
from src.retrieval.reranker import SemanticReranker

# Get documents
retriever = SIBILSRetriever()
docs = retriever.retrieve("What is malaria?", n=20)

# Rerank
reranker = SemanticReranker()
reranked = reranker.rerank("What is malaria?", docs, top_k=10)

print(f"Reranked {len(docs)} → {len(reranked)} documents")
print(f"Top doc: {reranked[0].title}")
EOF
```

### 4. Test Relevance Filter

```bash
./venv/bin/python3 << 'EOF'
from src.retrieval import SIBILSRetriever
from src.retrieval.relevance_filter import FastRelevanceFilter

# Get documents
retriever = SIBILSRetriever()
docs = retriever.retrieve("What is malaria?", n=20)

# Filter
filter = FastRelevanceFilter(min_overlap=0.15)
filtered = filter.filter_relevant("What is malaria?", docs, max_docs=10)

print(f"Filtered {len(docs)} → {len(filtered)} documents")
EOF
```

## Test Full V2 Pipeline

### Option 1: Start API Server

```bash
cd /home/egaillac/BioMoQA-RAG

# Clear GPU
nvidia-smi | grep VLLM | awk '{print $5}' | xargs -r kill -9

# Start V2 server
./venv/bin/python3 -m uvicorn api_server_v2:app --host 0.0.0.0 --port 9000
```

Then test at: http://egaillac.lan.text-analytics.ch:9000/docs

### Option 2: Python Script

```bash
./venv/bin/python3 << 'EOF'
from src.pipeline_vllm_v2 import EnhancedRAGPipeline, RAGConfigV2

config = RAGConfigV2(
    retrieval_n=50,
    use_query_expansion=True,
    use_reranking=True,
    use_relevance_filter=True,
    relevance_filter_type='fast',
    final_n=10,
    model_name='Qwen/Qwen2.5-7B-Instruct',
    use_vllm=True,
    gpu_memory_utilization=0.8,
)

pipeline = EnhancedRAGPipeline(config)

result = pipeline.run('What is AG1-IA?', debug=True)

print(f"\nQuestion: {result['question']}")
print(f"Time: {result['pipeline_time']}s")
print(f"Docs: {result['num_retrieved']}")
print(f"\nAnswer: {' '.join([s['text'] for s in result['answer']])[:200]}...")

if 'debug_info' in result:
    print(f"\nDebug Info:")
    for k, v in result['debug_info'].items():
        print(f"  {k}: {v}")
EOF
```

## Test V1 vs V2 Comparison

```bash
# Full comparison test (takes ~5-10 minutes)
./venv/bin/python3 test_v2_improvements.py
```

## Troubleshooting

### Import Error: No module named 'vllm'

**Solution:** Use venv python:
```bash
./venv/bin/python3 your_script.py
# NOT: python3 your_script.py
```

### GPU Out of Memory

**Solution:** Kill old processes:
```bash
nvidia-smi | grep VLLM | awk '{print $5}' | xargs -r kill -9
```

### Reranker Model Download Slow

First time loading the cross-encoder will download ~90MB model. This is normal.

## Requirements

All packages are in `requirements.txt`. To install:

```bash
./venv/bin/pip install -r requirements.txt
```

Key packages:
- `vllm` - Fast LLM inference
- `sentence-transformers` - For reranking
- `fastapi` - API server
- `transformers` - HuggingFace models
- `torch` - PyTorch

---

**Quick verification:**
```bash
./venv/bin/python3 -c "import vllm, sentence_transformers, fastapi; print('✓ Ready')"
```
