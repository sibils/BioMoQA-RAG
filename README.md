# BioMoQA-Ragnarök: Modern RAG for Biomedical Question Answering

## Overview
A research implementation of the Ragnarök RAG framework for biodiversity and biomedical question answering.

## Architecture

```
User Question → (R) Retrieval → Reranking → (AG) Generation → Structured Output
                     ↓              ↓              ↓
                 SIBILS API    LLM Reranker   Open-source LLM
                 (100 docs)    (Top 20)       (Citations)
```

## Components

### 1. Retrieval Module (`src/retrieval/`)
- SIBILS API integration (biodiversitypmc.sibils.org)
- Retrieves top-100 documents from PMC, PubMed, Plazi
- Elasticsearch-based biomedical search

### 2. Reranking Module (`src/reranking/`)
- Open-source LLM-based reranking
- Filters top-100 → top-20 most relevant
- Options: BGE-reranker, sentence-transformers

### 3. Generation Module (`src/generation/`)
- Open-source LLMs (Llama 3.1, Qwen, Mistral)
- Sentence-level citations
- JSON-structured output (Ragnarök format)

### 4. Evaluation Module (`src/evaluation/`)
- ROUGE-L, BERTScore, Exact Match
- Citation accuracy metrics
- Comprehensive benchmarking

## Project Structure

```
BioMoQA-Ragnarok/
├── src/
│   ├── retrieval/          # SIBILS API integration
│   ├── reranking/          # Document reranking
│   ├── generation/         # LLM-based answer generation
│   ├── evaluation/         # Metrics and evaluation
│   └── pipeline.py         # End-to-end RAG pipeline
├── data/
│   ├── questions/          # Test questions (120 QA pairs)
│   └── outputs/            # Generated answers
├── results/
│   ├── metrics/            # Evaluation results
│   └── analysis/           # Analysis notebooks
├── configs/
│   └── models.yaml         # Model configurations
└── notebooks/              # Research notebooks
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers sentence-transformers
pip install vllm accelerate bitsandbytes
pip install datasets evaluate rouge-score bert-score
pip install requests pandas numpy tqdm
```

## Usage

### Quick Start (Prototype)
```python
from src.pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    retrieval_n=100,
    rerank_n=20,
    model_name="meta-llama/Llama-3.1-8B-Instruct"
)

# Run question
result = rag.run("What is the host of Plasmodium falciparum?")
print(result['answer'])
```

### Full Evaluation
```bash
python -m src.evaluation.evaluate_all \
    --input data/questions/biomoqa_120.csv \
    --output results/metrics/baseline.json
```

## Models

### Open-Source Options
1. **Generation**: Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-7B-Instruct
2. **Reranking**: BAAI/bge-reranker-large, cross-encoder/ms-marco-MiniLM-L-12-v2
3. **Embeddings**: sentence-transformers/all-mpnet-base-v2

## Roadmap

- [x] Project setup
- [ ] Retrieval module (SIBILS)
- [ ] Reranking module
- [ ] Generation module (Llama 3.1)
- [ ] Evaluation framework
- [ ] Baseline experiments (120 QA)
- [ ] Scale to larger datasets
- [ ] Multi-model comparisons

## Citation

Based on:
- Ragnarök framework (Pradeep et al., 2024)
- TREC 2024 RAG Track guidelines
