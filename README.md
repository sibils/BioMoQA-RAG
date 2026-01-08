# BioMoQA RAG System

A high-performance biomedical question-answering system using Retrieval-Augmented Generation (RAG) with intelligent query parsing, hybrid retrieval, and optimized inference.

## Overview

This system answers biomedical questions by:
1. **Parsing queries** with SIBILS query parser (generates Elasticsearch queries with concept annotations)
2. **Retrieving documents** using hybrid search (Elasticsearch + Dense FAISS)
3. **Reranking** with cross-encoder semantic matching
4. **Generating answers** with vLLM-optimized LLM (Qwen 2.5-7B)
5. **Formatting** with sentence-level citations (Ragnarok-style)

## Performance

- **Speed**: 5-7 seconds per question (36x faster than baseline 177s)
- **Accuracy**: Comprehensive answers with sentence-level citations
- **Scalability**: Handles concurrent requests with FastAPI

## Quick Start

### Running the API

```bash
# API is already running on http://172.30.120.7:9000

# Check health
curl http://172.30.120.7:9000/health

# Ask a question
curl -X POST http://172.30.120.7:9000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes malaria?"}'
```

### API Response Format

```json
{
  "question": "What causes malaria?",
  "answer": [
    {
      "text": "Malaria is primarily caused by protozoan parasites.",
      "citation_ids": [3, 9],
      "citations": [
        {
          "document_id": 3,
          "document_title": "...",
          "pmcid": "PMCPMC11834219"
        }
      ]
    }
  ],
  "references": ["[0] PMC...: Title", ...],
  "pipeline_time": 6.98
}
```

## Key Features

### 1. SIBILS Query Parser Integration
- Automatically enhances queries with biomedical concepts
- Annotates with MeSH, NCIT, and AGROVOC ontologies
- Generates optimized Elasticsearch queries with concept expansion
- Removes punctuation-only clauses for clean queries

**Example:**
```
Input:  "What causes malaria?"
Output: Elasticsearch query with:
  - Text match: "causes" AND "malaria"
  - Concept expansion: mesh:D008288, ncit:C34797, agrovoc:c_34312
```

### 2. Hybrid Retrieval
- **Elasticsearch (SIBILS)**: Concept-aware search with ontology expansion
- **Dense (FAISS)**: Semantic vector search
- **Parallel**: Both run simultaneously
- **RRF Fusion**: Combines results intelligently

### 3. Ragnarok-Style Citations
- Sentence-level citations
- Structured JSON format
- Clean, readable text

## Architecture

```
User Question
     ↓
SIBILS Query Parser (ES query + concept annotations)
     ↓
Parallel Hybrid Retrieval (Elasticsearch + Dense)
     ↓
Cross-Encoder Reranking
     ↓
vLLM Generation (Qwen 2.5-7B)
     ↓
Citation Parsing
     ↓
Structured Response
```

## Project Structure

```
BioMoQA-RAG/
├── api_server_v3_fast.py          # FastAPI server (V3.1)
├── src/
│   ├── pipeline_vllm_v3_fast.py   # Main RAG pipeline
│   ├── retrieval/
│   │   ├── query_parser.py        # SIBILS query parser
│   │   ├── sibils_retriever.py    # BM25 with query parsing
│   │   ├── dense_retriever.py     # FAISS dense retrieval
│   │   ├── parallel_hybrid.py     # Hybrid orchestration
│   │   └── reranker.py            # Cross-encoder reranking
│   └── generation/
│       └── llm_generator.py       # vLLM generation
├── data/
│   ├── faiss_index.bin            # Dense index
│   └── documents.pkl              # 2398 documents
└── WORK_EXPLANATION.md            # Detailed project evolution
```

## Pipeline Evolution

| Version | Time | Key Innovation |
|---------|------|----------------|
| Baseline | 177s | Standard processing |
| V1 | 7.27s | vLLM optimization (24x faster) |
| V2 | 11.19s | Hybrid retrieval |
| V3 | 6.81s | Parallel + smart strategy |
| V3.1 | 6.98s | Query parser + citations |
| **V3.2** | **~7s** | **Full Elasticsearch queries with concept expansion** |

## Query Parser Flow

The query parser generates optimized Elasticsearch queries:

1. **Parse**: Call SIBILS query parser API with user question
2. **Generate ES Query**: Creates structured Elasticsearch query with:
   - Text matching clauses for keywords
   - Concept expansion (MeSH, NCIT, AGROVOC annotations)
   - Boolean logic (must/should clauses)
3. **Clean**: Remove punctuation-only clauses from ES query
4. **Retrieve**: POST ES query to SIBILS search API via `jq` parameter
5. **Fallback**: If ES query fails, fall back to keywords mode

## API Endpoints

### POST /qa
Ask a question:
```json
{
  "question": "What causes malaria?",
  "retrieval_n": 20,  // optional
  "final_n": 10,      // optional
  "debug": false      // optional
}
```

### GET /health
Check API status

### GET /docs
Interactive API documentation

## Configuration

Edit `api_server_v3_fast.py`:

```python
config = RAGConfigV3Fast(
    retrieval_n=20,              # Documents to retrieve
    use_smart_retrieval=True,    # Adaptive strategy
    use_reranking=True,          # Cross-encoder
    final_n=10,                  # Final context size
    gpu_memory_utilization=0.4   # GPU memory (adjust as needed)
)
```

## Performance Tips

### GPU Memory
- **MIG GPUs**: Use 0.4 (< 40GB)
- **Full A100**: Use 0.8 (< 64GB)

### Speed
- Reduce `retrieval_n` to 10
- Reduce `final_n` to 5
- Disable query parser (less accurate)

### Accuracy
- Increase `retrieval_n` to 50
- Increase `final_n` to 15
- Keep query parser enabled

## Troubleshooting

### API Won't Start
```bash
# Check GPU
nvidia-smi

# Kill old processes
nvidia-smi | grep python | awk '{print $5}' | xargs kill -9

# Check logs
tail -f logs/v3_api.log
```

### Clear GPU Memory
```bash
nvidia-smi | grep python | awk '{print $5}' | xargs kill -9
```

## Documentation

- [README.md](README.md) - This file
- [WORK_EXPLANATION.md](WORK_EXPLANATION.md) - Detailed evolution and design decisions
- [EVALUATION_REPORT.md](EVALUATION_REPORT.md) - Performance benchmarks

## Acknowledgments

- **SIBILS**: Query parser and BM25 retrieval API
- **vLLM**: Fast LLM inference
- **Qwen**: Open-source LLM from Alibaba
- **FAISS**: Dense vector search from Meta
