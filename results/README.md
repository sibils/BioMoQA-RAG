# Results Directory

This directory contains evaluation results from the BioMoQA RAG system.

## Files

### Generated Results
- `biomoqa_120_results.csv` - Full results for all 120 QA pairs
- `biomoqa_120_results.json` - Same results in JSON format
- `evaluation_detailed.csv` - Results with computed metrics (F1, ROUGE scores)
- `evaluation_summary.txt` - Quick summary statistics

### Column Explanations

#### biomoqa_120_results.csv

| Column | Description |
|--------|-------------|
| `question_id` | Sequential ID (1-120) |
| `question` | The biomedical question |
| `golden_answer` | Ground truth answer from dataset |
| `model_answer` | RAG-generated answer |
| `gold_context` | Original context from dataset |
| `top_retrieved_context` | Top 5 documents retrieved by SIBILS |
| `citations` | Comma-separated document IDs (0-based index) that support the answer |
| `pipeline_time_seconds` | Total time to process question |
| `num_documents_retrieved` | Number of documents used for generation (typically 20) |
| `response_length_chars` | Character count of generated answer |

#### Citation Format

**Example:** `citations` = "0, 4, 8, 17, 19"

This means the answer is supported by documents at positions 0, 4, 8, 17, and 19 in the retrieved document list.

To see which papers these refer to:
1. Check the `top_retrieved_context` column for document titles and PMCIDs
2. Or use the API with `include_documents=True` to get full citation details:

```python
response = requests.post(
    "http://localhost:9000/qa",
    json={"question": "Your question", "include_documents": True}
)

# Each answer sentence includes explicit citations:
for sentence in response.json()["answer"]:
    print(sentence["text"])
    for citation in sentence["citations"]:
        print(f"  - [{citation['document_id']}] {citation['document_title']} ({citation['pmcid']})")
```

## Regenerating Results

To regenerate the results:

```bash
# Method 1: Via API (requires server running)
python process_120_qa_via_api.py

# Method 2: Direct pipeline (slower, but doesn't need server)
python process_120_qa.py
```

## Evaluation

To run evaluation metrics:

```bash
python evaluate_results.py
```

This generates:
- Console output with summary statistics
- `evaluation_detailed.csv` with F1, ROUGE scores
- `evaluation_summary.txt` with key metrics

See [EVALUATION_REPORT.md](../EVALUATION_REPORT.md) for full analysis.

---

**Note:** CSV/JSON files are gitignored. Results must be regenerated locally.
