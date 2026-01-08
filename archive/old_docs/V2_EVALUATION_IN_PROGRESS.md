# V2 Evaluation In Progress

## Status

**Started:** 2026-01-07 15:05:00
**Process ID:** bd69b26
**Dataset:** 120 QA pairs from BioMoQA test set
**API Endpoint:** http://localhost:9000/qa (V2)

## Evaluation Configuration

- **Retrieval:** 100 documents (multi-query expansion)
- **Reranking:** Cross-encoder semantic reranking to top 30
- **Filtering:** Fast relevance filter to top 20
- **Model:** Qwen/Qwen2.5-7B-Instruct via vLLM
- **Debug mode:** Enabled (captures pipeline metrics)

## Expected Runtime

- **Estimated time:** 30-35 minutes
- **Per question:** ~15-16 seconds
- **Total questions:** 120

## Output Files

When complete:
- `results/biomoqa_120_v2_results.csv` - All results with answers and metrics
- `results/biomoqa_120_v2_errors.csv` - Any failed questions (if any)

## Next Steps

After completion:

1. **Calculate metrics:**
   ```bash
   ./venv/bin/python3 evaluate_results.py
   ```

2. **Compare with V1:**
   - V1 results: `results/biomoqa_120_results.csv`
   - V2 results: `results/biomoqa_120_v2_results.csv`
   - Expected improvements:
     - ROUGE-1: 40% → 60-70%
     - ROUGE-2: 18% → 30-40%
     - Pipeline time: 7s → 10-15s

3. **Analyze debug information:**
   - Query expansion effectiveness
   - Reranking impact on document selection
   - Relevance filtering statistics

## Monitoring

To check progress:
```bash
cat /tmp/claude/-home-egaillac/tasks/bd69b26.output
```

To check API health:
```bash
curl http://localhost:9000/health
```

## Background

This evaluation implements **Phase 1** of the improvements roadmap:
- ✅ Measure V2 actual performance
- ⏳ Compare with V1 baseline (40% ROUGE-1)
- ⏳ Identify where V2 still fails
- ⏳ Decide on Phase 2 improvements
