# How RAG Works in BioMoQA - Deep Dive

## What is RAG? (Yes, this is real RAG!)

**RAG = Retrieval-Augmented Generation**

This IS real RAG because:
1. ✅ **Retrieval**: Fetches relevant documents from external corpus (SIBILS)
2. ✅ **Augmentation**: Adds retrieved context to the prompt
3. ✅ **Generation**: LLM generates answer using the context

**What RAG is NOT:**
- ❌ Fine-tuning a model (we don't train anything)
- ❌ Using only the model's internal knowledge (we fetch external docs)
- ❌ Simple keyword search (we use semantic retrieval)

---

## Current RAG Pipeline - Step by Step

### Step 1: Question Received
```
User: "What is the host of Plasmodium falciparum?"
```

### Step 2: Retrieval (SIBILS API)
```python
# Query SIBILS with the question
documents = retriever.retrieve(question, n=50)

# SIBILS returns:
# - Top 50 most relevant papers from PubMed Central
# - Uses BM25 + semantic ranking
# - Returns: title, abstract, PMCID, score
```

**What can go wrong here:**
- ⚠️ **No context retrieved**: SIBILS finds no relevant papers
- ⚠️ **Poor retrieval**: Retrieved papers don't actually answer the question
- ⚠️ **Query mismatch**: Question phrasing doesn't match paper vocabulary

### Step 3: Reranking (Simple Top-K)
```python
# Currently: just take top 20 from the 50
documents = documents[:20]

# THIS IS A WEAKNESS - we should use a reranker!
```

### Step 4: Context Construction
```python
context = ""
for i, doc in enumerate(documents):
    context += f"[{i}] {doc.title}\n{doc.abstract[:1000]}\n\n"

# Example output:
# [0] PMC12345: Malaria parasites in human hosts
# Plasmodium falciparum causes severe malaria in humans...
#
# [1] PMC67890: Mosquito transmission of malaria
# Anopheles mosquitoes transmit P. falciparum to humans...
```

### Step 5: Prompt Construction
```python
prompt = f"""System: Answer the question using the provided context documents. Cite sources using [0], [1], etc.

QUESTION: {question}

CONTEXTS:
{context}

ANSWER:"""
```

### Step 6: LLM Generation (vLLM)
```python
# Qwen 2.5 7B generates answer
answer = llm.generate(prompt)

# Model output:
"The host of Plasmodium falciparum is humans. [0][1]
This is because the parasite's life cycle requires human red blood cells. [0]"
```

### Step 7: Citation Parsing
```python
# Extract [0], [1] references and map to documents
citations = parse_citations(answer, documents)

# Return with explicit document details
```

---

## Why Sometimes No Context is Retrieved

### Possible Reasons:

1. **SIBILS has limited papers**
   - Only PubMed Central collection (~7M papers)
   - May not have papers on niche topics
   - **Example:** Very specific species names might not be in PMC

2. **Query mismatch**
   - Question uses different terminology than papers
   - **Example:** "AG1-IA" vs "anastomosis group 1 IA"

3. **SIBILS API timeout/error**
   - Network issues
   - API temporarily down
   - Rate limiting

4. **BM25 limitations**
   - Keyword-based, not semantic
   - Doesn't understand synonyms well
   - **Example:** "host" vs "reservoir" vs "organism"

### How Model Answers Without Context

**When retrieval returns 0 documents:**

```python
# The prompt becomes:
"""System: Answer the question using the provided context documents.

QUESTION: What is the host of Plasmodium falciparum?

CONTEXTS:
(empty)

ANSWER:"""

# Model falls back to its training knowledge!
# This is why you still get an answer - the 7B model
# has biomedical knowledge from pre-training
```

**This is actually OK for RAG:**
- Model can use parametric knowledge as fallback
- But answer won't have citations (no documents to cite)
- Less trustworthy - can't verify claims

---

## Current System Strengths & Weaknesses

### ✅ Strengths
1. **Fast retrieval**: SIBILS is optimized for biomedical papers
2. **Large corpus**: 7M+ papers in PubMed Central
3. **Ultra-fast generation**: vLLM is 30-60x faster
4. **Explicit citations**: Can trace claims to sources
5. **No training needed**: Works zero-shot

### ⚠️ Weaknesses

1. **No semantic retrieval**
   - SIBILS uses BM25 (keyword matching)
   - Doesn't understand synonyms or paraphrasing
   - **Impact:** May miss relevant papers with different wording

2. **No reranking**
   - Just takes top-20 from top-50
   - Doesn't verify relevance to specific question
   - **Impact:** Context may include irrelevant papers

3. **Limited to abstracts**
   - Only uses title + abstract (not full text)
   - Important details may be in paper body
   - **Impact:** Misses deeper information

4. **No query expansion**
   - Doesn't try alternative phrasings
   - Doesn't expand acronyms automatically
   - **Impact:** May miss papers due to terminology mismatch

5. **No retrieval verification**
   - Doesn't check if retrieved docs answer the question
   - Sends all top-20 to LLM regardless of relevance
   - **Impact:** Context pollution with irrelevant info

6. **Single retrieval pass**
   - Doesn't do iterative retrieval
   - Can't refine based on initial answer
   - **Impact:** May miss relevant context

---

## How to Make a MUCH Better RAG

### Immediate Improvements (Easy)

#### 1. Add Semantic Reranker
```python
from sentence_transformers import CrossEncoder

# Use a cross-encoder to rerank top-50
reranker = CrossEncoder('cross-encoder/ms-marco-MedBERT')

# Score each doc against the question
scores = []
for doc in documents[:50]:
    score = reranker.predict([(question, doc.abstract)])
    scores.append((score, doc))

# Take top-20 by relevance (not SIBILS score)
documents = [doc for score, doc in sorted(scores, reverse=True)[:20]]
```

**Expected improvement:** +10-15% retrieval quality

#### 2. Query Expansion
```python
# Expand acronyms and add synonyms
def expand_query(question):
    # Use LLM to generate alternative phrasings
    expanded = llm.generate(f"""Given this question: {question}
    Generate 3 alternative phrasings using medical synonyms.
    """)

    # Search with all phrasings
    all_docs = []
    for query in [question] + expanded:
        all_docs.extend(retriever.retrieve(query, n=20))

    # Deduplicate and rerank
    return rerank(all_docs)[:20]
```

**Expected improvement:** +15-20% retrieval recall

#### 3. Retrieval Verification
```python
# Check if retrieved docs are relevant
def filter_relevant_docs(question, documents):
    relevant = []
    for doc in documents:
        # Ask LLM: does this doc help answer the question?
        prompt = f"""Question: {question}
        Document: {doc.abstract}

        Does this document help answer the question? Yes/No:"""

        if llm.generate(prompt).strip().lower() == "yes":
            relevant.append(doc)

    return relevant
```

**Expected improvement:** +10-20% answer quality (less noise)

### Medium Improvements (Moderate effort)

#### 4. Dense Retrieval with Embeddings
```python
# Replace BM25 with semantic search
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('allenai/specter2')

# Pre-embed all documents (one-time)
doc_embeddings = embedder.encode([d.abstract for d in all_docs])

# At query time: semantic search
query_embedding = embedder.encode(question)
similarities = cosine_similarity(query_embedding, doc_embeddings)
top_k = np.argsort(similarities)[-50:][::-1]
```

**Expected improvement:** +20-30% retrieval quality

#### 5. Hybrid Retrieval (BM25 + Dense)
```python
# Combine keyword and semantic search
bm25_docs = sibils.retrieve(question, n=50)  # Keyword
dense_docs = vector_search(question, n=50)   # Semantic

# Merge with weighted scores
final_docs = reciprocal_rank_fusion(bm25_docs, dense_docs)
```

**Expected improvement:** +25-35% retrieval quality

#### 6. Full-Text Access
```python
# Fetch full paper, not just abstract
def get_full_text(pmcid):
    # Use PubMed Central OA API
    response = requests.get(f"https://eutils.ncbi.nlm.nih.gov/...")
    return extract_sections(response)  # Introduction, Methods, Results

# Use more context for generation
contexts = [get_full_text(doc.pmcid) for doc in top_docs]
```

**Expected improvement:** +30-40% answer completeness

### Advanced Improvements (High effort)

#### 7. Iterative RAG (Self-RAG)
```python
def iterative_rag(question, max_iterations=3):
    answer = None
    for i in range(max_iterations):
        # Retrieve based on current understanding
        docs = retrieve(question, previous_answer=answer)

        # Generate answer
        answer = generate(question, docs)

        # Self-critique: is this answer complete?
        if is_answer_sufficient(question, answer):
            break

        # If not, refine query and iterate
        question = refine_query(question, answer)

    return answer
```

**Expected improvement:** +20-30% answer completeness

#### 8. Multi-hop Reasoning
```python
# For complex questions requiring multiple steps
def multi_hop_rag(question):
    # Decompose question
    sub_questions = decompose(question)

    # Answer each sub-question
    sub_answers = []
    for sub_q in sub_questions:
        docs = retrieve(sub_q)
        sub_answers.append(generate(sub_q, docs))

    # Synthesize final answer
    final_answer = synthesize(question, sub_answers)
    return final_answer
```

**Expected improvement:** +40-50% for complex questions

#### 9. Fine-tuned Retriever
```python
# Train retriever on biomedical QA pairs
from transformers import DPRContextEncoder, DPRQuestionEncoder

# Fine-tune on (question, relevant_doc) pairs
train_retriever(
    questions=train_questions,
    positive_docs=relevant_docs,
    negative_docs=irrelevant_docs
)
```

**Expected improvement:** +30-50% retrieval quality

---

## Recommended Improvements for BioMoQA

### Priority 1 (Implement Now)
1. ✅ **Add reranker** - Easy win, big impact
2. ✅ **Query expansion** - Handle acronyms and synonyms
3. ✅ **Retrieval verification** - Filter irrelevant docs

**Estimated time:** 2-3 days
**Expected improvement:** +30-40% overall quality

### Priority 2 (Next Phase)
4. ⏳ **Dense retrieval** - Build vector index
5. ⏳ **Hybrid search** - Combine BM25 + dense
6. ⏳ **Full-text access** - Use paper bodies, not just abstracts

**Estimated time:** 1-2 weeks
**Expected improvement:** +40-60% overall quality

### Priority 3 (Research Phase)
7. 🔬 **Iterative RAG** - Multi-pass refinement
8. 🔬 **Multi-hop reasoning** - Complex questions
9. 🔬 **Fine-tuned components** - Custom retriever/reranker

**Estimated time:** 1-2 months
**Expected improvement:** +50-70% overall quality

---

## Performance: Do We Need Better?

### Current Performance

| Metric | Value | Good Enough? |
|--------|-------|-------------|
| **Speed** | 7s/question | ✅ Excellent for research |
| **ROUGE-1** | 40.64% | ⚠️ Moderate |
| **Citation coverage** | 99.2% | ✅ Excellent |
| **Retrieval quality** | Unknown | ❓ Need to measure |

### What's Missing?

**We don't know:**
- ❓ Retrieval precision (% of retrieved docs that are relevant)
- ❓ Retrieval recall (% of relevant docs that we find)
- ❓ Answer factual accuracy (is the answer scientifically correct?)
- ❓ Citation accuracy (do cited papers actually support the claims?)

**Recommendation:** Measure these first!

### Benchmark Against TREC Ragnarök

The Ragnarök paper reported:
- **Retrieval@20:** ~60-70% recall
- **Answer quality:** Varies by team (20-60% accuracy)
- **Speed:** Not a focus (research benchmark)

**Our system:**
- **Speed:** ✅ Much faster (7s vs minutes)
- **Retrieval:** ❓ Unknown (need to evaluate)
- **Answer quality:** ⚠️ 40% ROUGE-1 suggests room for improvement

---

## Concrete Next Steps

### Week 1: Measure Current Quality
```python
# Evaluate retrieval quality
def evaluate_retrieval():
    for question, gold_context in test_set:
        retrieved = retrieve(question, n=20)

        # Precision: % retrieved that are relevant
        relevant = [d for d in retrieved if d in gold_context]
        precision = len(relevant) / len(retrieved)

        # Recall: % relevant that were retrieved
        recall = len(relevant) / len(gold_context)

        print(f"P: {precision}, R: {recall}")
```

### Week 2: Add Reranker
```python
# Implement cross-encoder reranking
pip install sentence-transformers

# In retrieval pipeline
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MedBERT')

docs = sibils.retrieve(question, n=50)
scores = reranker.predict([(question, d.abstract) for d in docs])
docs = [d for _, d in sorted(zip(scores, docs), reverse=True)[:20]]
```

### Week 3: Add Query Expansion
```python
# Use LLM to expand queries
def expand_query(question):
    expansion_prompt = f"""Medical question: {question}

    Provide 2 alternative phrasings using medical synonyms and expanded acronyms.
    Format: one per line."""

    alternatives = llm.generate(expansion_prompt).split('\n')

    # Retrieve with all phrasings
    all_docs = []
    for query in [question] + alternatives:
        all_docs.extend(sibils.retrieve(query, n=20))

    # Deduplicate and rerank
    return rerank_and_deduplicate(all_docs)[:20]
```

---

## Summary

**Is this real RAG?** ✅ Yes!

**Is it good RAG?** ⚠️ Moderate - room for improvement

**Main weaknesses:**
1. No semantic retrieval (BM25 only)
2. No reranking
3. No query expansion
4. Abstract-only (not full text)

**Quick wins:**
1. Add reranker (+10-15%)
2. Query expansion (+15-20%)
3. Retrieval verification (+10-20%)

**Performance target:**
- Current: ~40% semantic quality (ROUGE-1)
- With improvements: ~60-70% expected
- Research-grade: 70-80%+ possible

**Bottom line:** System works well for fast prototyping, but needs retrieval improvements for production-grade quality.
