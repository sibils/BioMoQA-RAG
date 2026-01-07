# Next Steps to Improve RAG Quality

Current V2 Status: **~60-70% expected quality (from 40% V1)**

## Phase 1: Evaluate V2 Performance (Do This First!)

Before adding more improvements, we need to measure V2's actual performance:

### 1.1 Run V2 on 120 QA Pairs

```bash
# Create V2 batch processor
./venv/bin/python3 << 'EOF'
# Save as: process_120_qa_v2.py

import pandas as pd
import requests
import time
from tqdm import tqdm

# Load dataset
df = pd.read_csv("/home/egaillac/Biomoqa/data/Question generation - biotXplorer - June 2024.csv")
df = df.iloc[2:, 4:].reset_index(drop=True)
df.columns = ["question", "golden_answer", "gold_context"]

# Process via V2 API
results = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        response = requests.post(
            "http://localhost:9000/qa",
            json={"question": row["question"], "debug": True},
            timeout=60
        )
        result = response.json()

        results.append({
            "question_id": idx + 1,
            "question": row["question"],
            "golden_answer": row["golden_answer"],
            "model_answer": " ".join([s["text"] for s in result["answer"]]),
            "gold_context": row["gold_context"],
            "pipeline_time": result["pipeline_time"],
            "pipeline_version": "v2",
            "debug_info": str(result.get("debug_info", {}))
        })

        time.sleep(0.5)
    except Exception as e:
        print(f"Error on question {idx+1}: {e}")

# Save
pd.DataFrame(results).to_csv("results/biomoqa_120_v2_results.csv", index=False)
print(f"✓ Processed {len(results)} questions")
EOF
```

### 1.2 Compare V1 vs V2 Metrics

```bash
./venv/bin/python3 evaluate_results.py  # Run on V2 results
# Compare ROUGE scores: V1 vs V2
```

**Expected V2 results:**
- ROUGE-1: 55-65% (up from 40%)
- ROUGE-2: 25-35% (up from 18%)
- Citation coverage: 99%+

---

## Phase 2: Hybrid Retrieval (Biggest Potential Gain)

**Current:** BM25 only (keyword matching)
**Upgrade:** BM25 + Dense embeddings (semantic matching)

**Expected improvement:** +25-35% retrieval quality

### 2.1 Build Dense Vector Index

```python
# In src/retrieval/dense_retriever.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class DenseRetriever:
    """Dense retrieval with SPECTER2 embeddings."""

    def __init__(self, model_name='allenai/specter2'):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents, save_path="data/dense_index.faiss"):
        """Build FAISS index from documents (one-time operation)."""
        print(f"Embedding {len(documents)} documents...")

        # Embed documents
        texts = [f"{d.title}. {d.abstract}" for d in documents]
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        # Build index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Save
        faiss.write_index(self.index, save_path)
        with open(save_path + ".docs.pkl", "wb") as f:
            pickle.dump(documents, f)

        print(f"✓ Index saved to {save_path}")

    def retrieve(self, question, k=50):
        """Retrieve top-k documents by semantic similarity."""
        query_embedding = self.embedder.encode([question])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        return [(self.documents[i], scores[0][j])
                for j, i in enumerate(indices[0])]
```

### 2.2 Implement Hybrid Search

```python
# In src/retrieval/hybrid_retriever.py

class HybridRetriever:
    """Combine BM25 (keyword) + Dense (semantic) retrieval."""

    def __init__(self, sibils_retriever, dense_retriever):
        self.sibils = sibils_retriever
        self.dense = dense_retriever

    def retrieve(self, question, k=50):
        """Hybrid retrieval with RRF (Reciprocal Rank Fusion)."""

        # Get from both retrievers
        bm25_docs = self.sibils.retrieve(question, n=100)
        dense_results = self.dense.retrieve(question, k=100)

        # RRF scoring
        scores = {}
        for rank, doc in enumerate(bm25_docs):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0) + 1/(rank + 60)

        for (doc, score), rank in zip(dense_results, range(len(dense_results))):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0) + 1/(rank + 60)

        # Merge and sort
        all_docs = {d.doc_id: d for d in bm25_docs + [d for d,s in dense_results]}
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda d: scores[d.doc_id],
            reverse=True
        )

        return sorted_docs[:k]
```

**Implementation time:** 2-3 days
**Expected gain:** +25-35%

---

## Phase 3: Better Reranker (Medical Domain)

**Current:** General cross-encoder (ms-marco)
**Upgrade:** Biomedical-specific reranker

### Options:

1. **MedCPT** (PubMed-trained)
   ```python
   reranker = CrossEncoder('ncbi/MedCPT-Cross-Encoder')
   ```

2. **BioLinkBERT** (Biomedical)
   ```python
   reranker = CrossEncoder('michiyasunaga/BioLinkBERT-large')
   ```

3. **PubMedBERT** (Fine-tuned)
   ```python
   from transformers import AutoModel, AutoTokenizer
   # Fine-tune microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
   ```

**Implementation time:** 1-2 days
**Expected gain:** +10-15%

---

## Phase 4: Full-Text Retrieval

**Current:** Abstract only (~300 words)
**Upgrade:** Full paper body (~5000 words)

### 4.1 Access PubMed Central Full Text

```python
import requests
from bs4 import BeautifulSoup

def get_fulltext(pmcid):
    """Fetch full text from PMC OA."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract sections
    sections = {}
    for section in soup.find_all('div', class_='tsec'):
        title = section.find('h2')
        if title:
            sections[title.text] = section.get_text()

    return sections
```

### 4.2 Chunk Long Documents

```python
def chunk_document(text, chunk_size=500, overlap=100):
    """Split document into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
```

**Implementation time:** 3-5 days
**Expected gain:** +30-40% for complex questions

---

## Phase 5: Iterative Refinement (Self-RAG)

**Current:** Single-pass retrieval
**Upgrade:** Multi-pass with self-critique

```python
def iterative_rag(question, max_iterations=2):
    """Iterative retrieval with self-assessment."""

    answer = None
    retrieved_docs = set()

    for i in range(max_iterations):
        # Retrieve (avoid duplicates)
        new_docs = retrieve(question, exclude=retrieved_docs)
        retrieved_docs.update(new_docs)

        # Generate answer
        answer = generate(question, new_docs)

        # Self-critique: Is this answer complete?
        critique = llm.generate(f"""
        Question: {question}
        Answer: {answer}

        Is this answer complete and fully addresses the question?
        If not, what information is missing?

        Reply: YES (complete) or NO: [what's missing]
        """)

        if "YES" in critique:
            break  # Answer is good

        # Refine query based on what's missing
        question = refine_query(question, critique)

    return answer
```

**Implementation time:** 2-3 days
**Expected gain:** +20-30% for complex questions

---

## Phase 6: Fine-Tune Components

### 6.1 Fine-Tune Retriever

Train on biomedical QA pairs:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Create training pairs
train_examples = [
    InputExample(texts=[question, relevant_doc], label=1.0),
    InputExample(texts=[question, irrelevant_doc], label=0.0),
    ...
]

# Fine-tune
model = SentenceTransformer('allenai/specter2')
train_dataloader = DataLoader(train_examples, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3
)
```

**Implementation time:** 1-2 weeks
**Expected gain:** +30-50% retrieval quality

---

## Recommended Priority

### Quick Wins (1-2 weeks)
1. ✅ **Evaluate V2** - Measure actual improvement
2. 🔄 **Hybrid retrieval** - Biggest gain for effort
3. 🔄 **Medical reranker** - Easy swap, good gain

### Medium Term (1 month)
4. Full-text retrieval
5. Better LLM (Qwen 2.5 14B or 32B)
6. Iterative refinement

### Long Term (2-3 months)
7. Fine-tune retriever
8. Fine-tune reranker
9. Custom training on biomedical QA

---

## Implementation Plan

### Week 1: Evaluation
- [ ] Run V2 on 120 QA pairs
- [ ] Calculate ROUGE scores
- [ ] Compare with V1 baseline
- [ ] Identify where V2 still fails

### Week 2: Hybrid Retrieval
- [ ] Download PMC papers for dense index
- [ ] Build FAISS index with SPECTER2
- [ ] Implement RRF fusion
- [ ] Test on 20 questions

### Week 3: Medical Reranker
- [ ] Try MedCPT, BioLinkBERT, PubMedBERT
- [ ] Compare on 50 questions
- [ ] Choose best model
- [ ] Deploy to V2

### Week 4: Full Evaluation
- [ ] Run improved V2 on 120 questions
- [ ] Calculate final metrics
- [ ] Write evaluation report

---

## Expected Final Performance

| Metric | V1 | V2 (Current) | V2 + Hybrid | V2 + Full-text |
|--------|-----|--------------|-------------|----------------|
| ROUGE-1 | 40% | 60-70% | 75-85% | 80-90% |
| ROUGE-2 | 18% | 30-40% | 45-55% | 50-65% |
| Time/Q | 7s | 8-12s | 10-15s | 12-18s |

**Goal:** Achieve 80%+ ROUGE-1 with hybrid retrieval + medical reranker

---

## Next Steps for You

1. **Measure V2 baseline:**
   ```bash
   # Run V2 on 120 QA pairs
   sudo systemctl start biomoqa-rag
   # Wait for API to start, then:
   ./venv/bin/python3 process_120_qa_v2.py
   ```

2. **Decide on next improvement:**
   - Hybrid retrieval (biggest gain)
   - Medical reranker (easiest)
   - Full-text (most comprehensive)

3. **I'll implement your choice!**

Which improvement should we tackle first?
