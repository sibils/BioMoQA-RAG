# BioMoQA-Ragnarök Quick Start Guide

## What You Have

A complete, modern RAG system for biodiversity/biomedical question answering based on the Ragnarök framework (TREC 2024).

### Architecture
```
Question → [SIBILS Retrieval] → [Reranking] → [Llama 3.1 Generation] → Answer + Citations
              (100 docs)           (top-20)         (8B, 4-bit)            (JSON format)
```

## Installation

```bash
cd /home/egaillac/BioMoQA-Ragnarok
source venv/bin/activate  # Virtual environment already created
```

Dependencies are already installed!

## Quick Test (Retrieval Only - No LLM Download)

```bash
python test_prototype.py
```

This tests:
- SIBILS API retrieval
- Document parsing
- Basic pipeline

## Full Test (Downloads Llama 3.1 8B - ~8GB)

```bash
python run_simple_test.py
```

This will:
1. Download Llama-3.1-8B-Instruct (~8GB, first time only)
2. Run full RAG pipeline on test question
3. Generate answer with citations
4. Save results to `results/simple_test_output.json`

**Note:** First run downloads the model, takes ~10-15 minutes. Subsequent runs are fast.

## Using the Pipeline in Code

```python
from src.pipeline import RAGPipeline, RAGConfig

# Configure pipeline
config = RAGConfig(
    retrieval_n=50,              # Retrieve 50 documents
    rerank_n=10,                 # Keep top-10
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,           # Memory efficient
)

# Initialize
pipeline = RAGPipeline(config)

# Run on a question
result = pipeline.run(
    "What is the host of Plasmodium falciparum?",
    topic_id="Q001"
)

# Access answer
for sentence in result['answer']:
    print(f"{sentence['text']} {sentence['citations']}")
```

## Output Format (Ragnarök Standard)

```json
{
  "topic_id": "Q001",
  "question": "What is the host of...",
  "references": ["doc0", "doc1", ...],
  "response_length": 728,
  "answer": [
    {
      "text": "The host of Plasmodium falciparum is humans.",
      "citations": [0, 1, 5]
    },
    {
      "text": "Transmission occurs through Anopheles mosquitoes.",
      "citations": [2, 3]
    }
  ],
  "pipeline_time": 12.5,
  "num_retrieved": 50
}
```

## Next Steps

### 1. Test on Your 120 QA Pairs

```bash
# Copy your dataset
cp ~/Biomoqa/data/Question*.csv data/questions/

# Run batch evaluation (coming next)
python -m src.evaluation.evaluate_batch \
    --input data/questions/biomoqa_120.csv \
    --output results/biomoqa_120_results.json
```

### 2. Try Different Models

Edit `run_simple_test.py` and change `model_name`:

**Small models (faster, less memory):**
- `meta-llama/Llama-3.1-8B-Instruct` (current, 8GB)
- `Qwen/Qwen2.5-7B-Instruct` (7GB)
- `mistralai/Mistral-7B-Instruct-v0.3` (7GB)

**Larger models (better quality, need more VRAM):**
- `meta-llama/Llama-3.1-70B-Instruct` (35GB with 4-bit)
- `Qwen/Qwen2.5-14B-Instruct` (14GB with 4-bit)

### 3. Scale Up Retrieval

```python
config = RAGConfig(
    retrieval_n=200,    # More documents
    rerank_n=30,        # More context for LLM
)
```

### 4. Add Reranking

Currently uses simple score-based filtering. To add real reranking:

```python
# TODO: Implement in src/reranking/reranker.py
# Options: BGE-reranker, cross-encoders, etc.
```

## System Resources

Your GPU: **A100 80GB** (excellent!)

Memory usage with Llama-3.1-8B-Instruct (4-bit):
- Model: ~5-6GB VRAM
- Inference: ~2-3GB VRAM
- Total: ~8-10GB VRAM (plenty of headroom)

You can run multiple models in parallel or use larger models (e.g., 70B).

## Project Structure

```
BioMoQA-Ragnarok/
├── src/
│   ├── retrieval/          # SIBILS API integration ✓
│   ├── generation/         # LLM with citations ✓
│   ├── pipeline.py         # End-to-end RAG ✓
│   ├── reranking/          # TODO: Advanced reranking
│   └── evaluation/         # TODO: Metrics (ROUGE, BERTScore, etc.)
├── data/
│   └── questions/          # Your 120 QA pairs
├── results/                # Output files
└── test_prototype.py       # Quick test
```

## Troubleshooting

### Out of Memory
```python
config = RAGConfig(load_in_4bit=True)  # Already enabled
# Or try smaller model: Qwen2.5-7B-Instruct
```

### Slow Generation
```python
# Reduce max tokens
config = RAGConfig(max_new_tokens=256)

# Or use vLLM for faster inference (advanced)
```

### SIBILS API Errors
```python
# Check internet connection
# The API is free and public, no auth needed
```

## What's Next?

The system is ready for research! Remaining todos:

1. **Evaluation Module**: ROUGE-L, BERTScore, exact match
2. **Reranking Module**: Cross-encoder or LLM-based reranking
3. **Batch Processing**: Run on all 120 questions
4. **Analysis**: Compare with your old results

Want me to implement any of these next?

## Contact

Based on:
- Ragnarök framework: https://github.com/castorini/ragnarok
- TREC 2024 RAG Track: https://trec-rag.github.io
