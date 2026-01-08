# BioMoQA-Ragnarök Server Status

## Server is LIVE!

**URL:** http://egaillac.lan.text-analytics.ch:9000
**Status:** Running on port 9000
**Network:** Accessible from VPN

## Quick Access

- **Health Check:** http://egaillac.lan.text-analytics.ch:9000/health
- **API Docs:** http://egaillac.lan.text-analytics.ch:9000/docs
- **QA Endpoint:** http://egaillac.lan.text-analytics.ch:9000/qa

## Test from Command Line

```bash
curl -X POST "http://egaillac.lan.text-analytics.ch:9000/qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the host of Plasmodium falciparum?"}'
```

## Test from Python

```python
import requests

response = requests.post(
    "http://egaillac.lan.text-analytics.ch:9000/qa",
    json={"question": "What is the host of Plasmodium falciparum?"}
)

print(response.json())
```

## Server Details

- **Model:** Qwen 2.5 7B Instruct
- **Engine:** vLLM (ultra-fast inference)
- **Retrieval:** SIBILS API (10,000+ biomedical papers)
- **Performance:** ~3-6 seconds per question
- **GPU:** A100 80GB

## Next Steps

1. Share this URL with colleagues on VPN
2. Process 120 QA pairs: `python process_120_qa.py`
3. Analyze results in `results/biomoqa_120_results.csv`

---

**Server started:** 2026-01-06 23:12
**Started by:** ./start_api.sh
