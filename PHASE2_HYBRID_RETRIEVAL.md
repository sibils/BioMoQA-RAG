# Phase 2: Hybrid Retrieval Implementation

## Overview

Phase 2 adds **hybrid retrieval** combining:
- **BM25 (SIBILS API)**: Keyword-based retrieval (existing)
- **Dense embeddings (FAISS)**: Semantic similarity search (new)
- **Reciprocal Rank Fusion (RRF)**: Combines both rankings

**Expected improvement**: +25-35% retrieval quality over V2

## Architecture

```
Question
    │
    ├─→ BM25 (SIBILS) ────→ Top 50 documents
    │                            │
    └─→ Dense (FAISS) ─────→ Top 50 documents
                                 │
                            ┌────┴────┐
                            │   RRF   │  Reciprocal Rank Fusion
                            │ Combine │  Score = Σ 1/(k + rank_i)
                            └─────────┘
                                 │
                            Top 20 fused results
                                 │
                            ┌─────────┐
                            │ Reranker│  Cross-encoder (V2)
                            └─────────┘
                                 │
                            Top 20 final documents
                                 │
                           [Generation]
```

## Components

### 1. Dense Retriever

**File**: `src/retrieval/dense_retriever.py`

```python
from src.retrieval.dense_retriever import DenseRetriever

# Initialize
dense = DenseRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load pre-built index
dense.load("data/faiss_index.bin", "data/documents.pkl")

# Retrieve documents
results = dense.retrieve("What causes malaria?", top_k=50)
```

**Features**:
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- FAISS `IndexFlatIP` for cosine similarity
- Pre-built index of ~2,000 biomedical documents from SIBILS

### 2. Hybrid Retriever

**File**: `src/retrieval/dense_retriever.py`

```python
from src.retrieval.hybrid_retriever import HybridRetriever

# Combine retrievers
hybrid = HybridRetriever(
    sibils_retriever=sibils,
    dense_retriever=dense,
    alpha=0.5,  # 50% BM25, 50% dense
    k=60        # RRF constant
)

# Retrieve with both methods
results = hybrid.retrieve("What causes malaria?", n=50, top_k=20)
```

**Parameters**:
- `alpha`: Weight for dense retrieval (0=BM25 only, 1=dense only, 0.5=balanced)
- `k`: RRF constant (60 is standard from original RRF paper)
- `n`: Documents to retrieve from each source
- `top_k`: Final number of documents after fusion

### 3. Reciprocal Rank Fusion (RRF)

RRF combines rankings from multiple sources:

```
RRF_score(doc) = Σ [1 / (k + rank_i)]
```

Where:
- `rank_i` is the document's rank in retrieval method i
- `k` is a constant (typically 60)
- Sum is over all methods that returned the document

**Benefits**:
- No need to normalize scores across different retrieval systems
- Robust to outliers
- Simple and effective

## Building the Dense Index

**Script**: `build_dense_index.py`

```bash
# Build index (takes 5-10 minutes)
./venv/bin/python3 build_dense_index.py
```

**What it does**:
1. Queries SIBILS with 24 broad biomedical topics
2. Collects ~2,000 unique documents
3. Generates embeddings for all documents
4. Builds FAISS index for fast similarity search
5. Saves to `data/faiss_index.bin` and `data/documents.pkl`

**Corpus coverage**:
- Diseases: cancer, infection, diabetes, cardiovascular, neurological, autoimmune
- Organisms: bacteria, viruses, fungi, parasites
- Molecular biology: proteins, genes, pathways, immune response
- Methods: clinical trials, diagnostics, vaccines, drugs

## Integration with V2 Pipeline

Hybrid retrieval integrates seamlessly with existing V2 improvements:

```python
# V2 + Phase 2 Pipeline
from src.retrieval.sibils_retriever import SIBILSRetriever
from src.retrieval.dense_retriever import DenseRetriever, HybridRetriever
from src.retrieval.reranker import SemanticReranker
from src.retrieval.relevance_filter import FastRelevanceFilter
from src.retrieval.query_expander import HybridQueryExpander

# Initialize components
sibils = SIBILSRetriever()
dense = DenseRetriever()
dense.load("data/faiss_index.bin", "data/documents.pkl")

# Hybrid retriever
hybrid = HybridRetriever(sibils, dense, alpha=0.5)

# V2 components
query_expander = HybridQueryExpander()
reranker = SemanticReranker()
filter = FastRelevanceFilter()

# Pipeline flow:
# 1. Expand query
expanded = query_expander.expand(question)

# 2. Retrieve with hybrid approach (50 from each)
all_docs = []
for query in expanded.all_queries:
    docs = hybrid.retrieve(query, n=50, top_k=20)
    all_docs.extend(docs)

# 3. Rerank (20 best)
reranked = reranker.rerank(question, all_docs, top_k=20)

# 4. Filter (final 20)
final_docs = filter.filter_relevant(question, reranked, max_docs=20)

# 5. Generate answer
answer = generate(question, final_docs)
```

## Expected Performance

### Retrieval Quality
- **V2 baseline**: ~14 relevant documents per query
- **Phase 2 (hybrid)**: ~17-19 relevant documents per query
- **Improvement**: +25-35% more relevant documents

### Why Hybrid Works Better

1. **Complementary strengths**:
   - BM25: Good for exact matches, acronyms, specific terms
   - Dense: Good for semantic similarity, paraphrases, concepts

2. **Example query**: "What causes malaria?"
   - BM25 finds: "malaria", "Plasmodium", "mosquito"
   - Dense finds: "parasitic disease", "tropical infection", "Anopheles transmission"

3. **Robustness**:
   - If one method fails (e.g., no exact keyword match), the other compensates
   - RRF naturally upweights documents found by both methods

### Pipeline Time
- **V2**: 11.19s per question
- **Phase 2**: 11.5-12s per question (+3-7%)
- **Tradeoff**: Minimal slowdown for significant quality gain

## Testing Phase 2

```bash
# Test hybrid retrieval
./venv/bin/python3 test_hybrid_retrieval.py

# Run full evaluation
./venv/bin/python3 process_120_qa_v2p2.py

# Compare V2 vs V2+Phase2
./venv/bin/python3 compare_v2_v2p2.py
```

## Model Options

### Dense Retrieval Models

Current: `sentence-transformers/all-MiniLM-L6-v2` (fast, general)

**Biomedical alternatives** (for Phase 3):
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`: Pretrained on PubMed
- `dmis-lab/biobert-base-cased-v1.2`: BioBERT for biomedical text
- `allenai/scibert_scivocab_uncased`: Scientific papers

**When to upgrade**:
- If Phase 2 shows good gains, switch to biomedical model
- Expected additional improvement: +10-15%

## Next Steps After Phase 2

1. **Evaluate gains**: Measure actual improvement on 120 QA pairs
2. **Tune alpha**: Experiment with different BM25/dense weights
3. **Consider Phase 3**: Medical-domain reranker (MedCPT, BioLinkBERT)
4. **Consider Phase 4**: Full-text retrieval (not just abstracts)

## Files Created

- `src/retrieval/dense_retriever.py` - Dense retrieval and hybrid fusion
- `build_dense_index.py` - Index building script
- `data/faiss_index.bin` - FAISS vector index
- `data/documents.pkl` - Document metadata
- `PHASE2_HYBRID_RETRIEVAL.md` - This documentation

## References

- **RRF Paper**: Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009)
- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
