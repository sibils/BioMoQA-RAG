# RAG Improvements Roadmap

## Quick Wins (2-3 days implementation)

### 1. Cross-Encoder Reranking

**Current:** Take top-20 from SIBILS top-50 (no verification)
**Improved:** Rerank with semantic relevance scoring

```bash
pip install sentence-transformers
```

```python
# In src/retrieval/reranker.py
from sentence_transformers import CrossEncoder

class SemanticReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MedMarco-electra-base'):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, documents: List[Document], top_k: int = 20):
        # Score each document against question
        pairs = [(question, doc.title + " " + doc.abstract) for doc in documents]
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(zip(scores, documents), reverse=True)
        return [doc for score, doc in ranked[:top_k]]
```

**Expected impact:** +10-15% better retrieval

---

### 2. Query Expansion

**Current:** Single query to SIBILS
**Improved:** Multiple query variants with acronym expansion

```python
# In src/retrieval/query_expander.py

class QueryExpander:
    def __init__(self, llm):
        self.llm = llm

    def expand(self, question: str) -> List[str]:
        prompt = f"""Given this biomedical question: "{question}"

Generate 2 alternative phrasings that:
1. Expand any acronyms (e.g., AG1-IA -> anastomosis group 1 IA)
2. Use medical synonyms (e.g., host -> reservoir organism)
3. Rephrase technically

Format: one per line, no numbering."""

        alternatives = self.llm.generate(prompt, max_tokens=100).strip().split('\n')
        return [question] + [alt.strip() for alt in alternatives if alt.strip()]

    def multi_query_retrieve(self, question: str, retriever, n_per_query=20):
        queries = self.expand(question)

        all_docs = []
        seen_ids = set()

        for query in queries:
            docs = retriever.retrieve(query, n=n_per_query)
            for doc in docs:
                if doc.doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc.doc_id)

        return all_docs
```

**Expected impact:** +15-20% better recall

---

### 3. Retrieval Verification Filter

**Current:** Send all top-20 to LLM
**Improved:** Filter to only relevant documents

```python
# In src/retrieval/relevance_filter.py

class RelevanceFilter:
    def __init__(self, llm):
        self.llm = llm

    def filter_relevant(self, question: str, documents: List[Document]) -> List[Document]:
        relevant_docs = []

        for doc in documents:
            prompt = f"""Question: {question}

Document title: {doc.title}
Document abstract: {doc.abstract[:500]}

Is this document relevant to answering the question?
Answer only: YES or NO"""

            response = self.llm.generate(prompt, max_tokens=5).strip().upper()

            if "YES" in response:
                relevant_docs.append(doc)

        return relevant_docs
```

**Expected impact:** +10-20% cleaner context

---

## Medium Improvements (1-2 weeks)

### 4. Hybrid Retrieval (BM25 + Dense)

Build a local dense retriever alongside SIBILS:

```python
# In src/retrieval/hybrid_retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class HybridRetriever:
    def __init__(self, sibils_retriever):
        self.sibils = sibils_retriever
        self.embedder = SentenceTransformer('allenai/specter2')
        self.index = None  # FAISS index
        self.documents = []  # Document store

    def build_index(self, documents: List[Document]):
        """One-time: embed all documents"""
        self.documents = documents
        embeddings = self.embedder.encode([d.abstract for d in documents])

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product
        self.index.add(embeddings)

    def dense_retrieve(self, question: str, k=50):
        query_embedding = self.embedder.encode([question])
        scores, indices = self.index.search(query_embedding, k)

        return [(self.documents[i], scores[0][j])
                for j, i in enumerate(indices[0])]

    def hybrid_retrieve(self, question: str, k=20):
        # BM25 from SIBILS
        bm25_docs = self.sibils.retrieve(question, n=50)

        # Dense retrieval
        dense_results = self.dense_retrieve(question, k=50)

        # Reciprocal Rank Fusion
        scores = {}
        for rank, doc in enumerate(bm25_docs):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0) + 1/(rank + 60)

        for (doc, score), rank in zip(dense_results, range(len(dense_results))):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0) + 1/(rank + 60)

        # Sort by combined score
        ranked_docs = sorted(bm25_docs + [d for d, s in dense_results],
                           key=lambda d: scores[d.doc_id], reverse=True)

        # Deduplicate
        seen = set()
        final_docs = []
        for doc in ranked_docs:
            if doc.doc_id not in seen:
                final_docs.append(doc)
                seen.add(doc.doc_id)
                if len(final_docs) >= k:
                    break

        return final_docs
```

**Expected impact:** +25-35% better retrieval

---

## Implementation Plan

### Phase 1: Add Reranking (1 day)
- [ ] Install sentence-transformers
- [ ] Create `src/retrieval/reranker.py`
- [ ] Integrate into `pipeline_vllm.py`
- [ ] Test on 10 sample questions
- [ ] Measure improvement

### Phase 2: Query Expansion (1 day)
- [ ] Create `src/retrieval/query_expander.py`
- [ ] Use existing vLLM for expansion
- [ ] Add to pipeline with config flag
- [ ] Test and measure

### Phase 3: Relevance Filtering (1 day)
- [ ] Create `src/retrieval/relevance_filter.py`
- [ ] Add optional filtering step
- [ ] Benchmark with/without
- [ ] Measure quality vs speed tradeoff

### Phase 4: Evaluation Suite (2 days)
- [ ] Create retrieval evaluation script
- [ ] Measure precision/recall
- [ ] Compare before/after improvements
- [ ] Document results

### Phase 5: Hybrid Retrieval (1 week)
- [ ] Build dense index (FAISS)
- [ ] Implement hybrid search
- [ ] Benchmark thoroughly
- [ ] Deploy if better

---

## Expected Results

| Improvement | Implementation Time | Expected Gain |
|------------|-------------------|---------------|
| Reranking | 1 day | +10-15% |
| Query expansion | 1 day | +15-20% |
| Relevance filter | 1 day | +10-20% |
| Hybrid retrieval | 1 week | +25-35% |
| **Total** | **2 weeks** | **+40-60%** |

**Current ROUGE-1:** 40.64%
**Expected after improvements:** 55-65% ROUGE-1

---

## Configuration Example

```python
# Updated RAGConfig
@dataclass
class RAGConfig:
    # Retrieval
    retrieval_n: int = 100
    use_query_expansion: bool = True
    use_reranking: bool = True
    use_relevance_filter: bool = False  # Optional (slower but cleaner)
    rerank_n: int = 20

    # Models
    reranker_model: str = "cross-encoder/ms-marco-MedMarco-electra-base"
    use_hybrid_retrieval: bool = False  # Future

    # Generation (unchanged)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_vllm: bool = True
```

---

Ready to implement? Start with Phase 1 (reranking) for the biggest quick win!
