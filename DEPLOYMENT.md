# BioMoQA-Ragnarök Deployment Guide

## 🚀 Quick Start

### 1. Start the API Server

```bash
cd /home/egaillac/BioMoQA-Ragnarok
./start_api.sh
```

**The API will be available at:**
- **Local:** http://localhost:7000
- **Network:** http://egaillac.lan.text-analytics.ch:7000
- **Docs:** http://egaillac.lan.text-analytics.ch:7000/docs

### 2. Test the API

**Using curl:**
```bash
curl -X POST "http://localhost:7000/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the host of Plasmodium falciparum?",
    "retrieval_n": 50,
    "rerank_n": 20
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://egaillac.lan.text-analytics.ch:7000/qa",
    json={
        "question": "What is the host of Plasmodium falciparum?",
        "retrieval_n": 50,
        "rerank_n": 20
    }
)

print(response.json())
```

**Using browser:**
Go to http://egaillac.lan.text-analytics.ch:7000/docs for interactive API documentation

---

## 📊 Process 120 QA Pairs

```bash
cd /home/egaillac/BioMoQA-Ragnarok
source venv/bin/activate
python process_120_qa.py
```

**Output:**
- `results/biomoqa_120_results.csv` - All results in CSV format
- `results/biomoqa_120_results.json` - JSON format for analysis

**CSV Columns:**
- `question_id` - Question number (1-120)
- `question` - The biomedical question
- `golden_answer` - Ground truth answer
- `model_answer` - RAG-generated answer
- `gold_context` - Original context from dataset
- `top_retrieved_context` - Top 5 documents retrieved by SIBILS
- `citations` - Document citations used
- `pipeline_time_seconds` - Time to generate answer
- `num_documents_retrieved` - Number of docs retrieved
- `response_length_chars` - Answer length

---

## ⚡ Performance Expectations

### With vLLM (Fast Mode):
- **Retrieval:** ~2-5 seconds (SIBILS API)
- **Generation:** ~0.5-1 second (vLLM)
- **Total:** ~3-6 seconds per question
- **120 questions:** ~6-12 minutes total

### Old System (HuggingFace Transformers):
- **Generation:** ~30-60 seconds
- **Total:** ~177 seconds per question (as we saw)
- **Speedup:** **30-60x faster with vLLM!**

---

## 🏗️ Project Structure

```
/home/egaillac/BioMoQA-Ragnarok/
│
├── api_server.py              # FastAPI server (port 7000)
├── process_120_qa.py          # Batch processing for 120 QA
├── start_api.sh               # Start server script
│
├── src/
│   ├── retrieval/             # SIBILS API integration
│   ├── generation/            # LLM generation
│   ├── pipeline.py            # Standard RAG pipeline
│   └── pipeline_vllm.py       # Fast pipeline with vLLM ⚡
│
├── scripts/
│   ├── test_prototype.py      # Test retrieval only
│   └── run_simple_test.py     # Test full pipeline
│
├── data/
│   ├── inputs/                # Input datasets
│   └── outputs/               # Generated outputs
│
├── results/
│   ├── biomoqa_120_results.csv    # Batch results
│   └── simple_test_output.json    # Single test output
│
├── logs/                      # Server logs
│
└── docs/                      # Documentation
    ├── README.md
    ├── QUICKSTART.md
    ├── SYSTEM_OVERVIEW.md
    └── DEPLOYMENT.md (this file)
```

---

## 🌐 API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check - returns model status

### `POST /qa`
Main QA endpoint

**Request:**
```json
{
  "question": "Your biomedical question",
  "retrieval_n": 50,
  "rerank_n": 20,
  "include_documents": false
}
```

**Response:**
```json
{
  "question": "What is the host of...",
  "answer": [
    {
      "text": "The host is humans.",
      "citations": [0, 1, 3]
    }
  ],
  "references": ["doc0", "doc1", ...],
  "response_length": 456,
  "pipeline_time": 3.2,
  "num_retrieved": 50
}
```

### `GET /stats`
Pipeline statistics and configuration

### `GET /docs`
Interactive API documentation (Swagger UI)

---

## 🔧 Advanced Configuration

### Run API in Background (Production)

```bash
# Using nohup
nohup ./start_api.sh > logs/api.log 2>&1 &

# Monitor logs
tail -f logs/api.log
```

### Run with Gunicorn (Production)

```bash
source venv/bin/activate
gunicorn api_server:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:7000 \
  --timeout 300 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Systemd Service (Auto-start on Boot)

Create `/etc/systemd/system/biomoqa-rag.service`:

```ini
[Unit]
Description=BioMoQA-Ragnarök API
After=network.target

[Service]
Type=simple
User=egaillac
WorkingDirectory=/home/egaillac/BioMoQA-Ragnarok
ExecStart=/home/egaillac/BioMoQA-Ragnarok/venv/bin/gunicorn api_server:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:7000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable biomoqa-rag
sudo systemctl start biomoqa-rag
sudo systemctl status biomoqa-rag
```

---

## 📈 Monitoring

### Check GPU Usage
```bash
nvidia-smi
```

### Check Server Status
```bash
curl http://localhost:7000/health
```

### View Logs
```bash
tail -f logs/api.log
```

---

## 🐛 Troubleshooting

### API won't start
```bash
# Check if port 7000 is already in use
lsof -i :7000

# Kill existing process
kill -9 $(lsof -t -i :7000)
```

### Out of GPU memory
```bash
# Reduce GPU memory utilization in api_server.py:
# gpu_memory_utilization=0.6  # Instead of 0.8
```

### Slow inference
```bash
# Make sure vLLM is enabled (use_vllm=True)
# Check GPU is being used:
nvidia-smi
```

---

## 🔐 Security (if exposing publicly)

### Add API Key Authentication

```python
# Add to api_server.py
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/qa")
async def answer_question(
    request: QuestionRequest,
    api_key: str = Depends(api_key_header)
):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of code
```

### Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/qa")
@limiter.limit("10/minute")
async def answer_question(...):
    ...
```

---

## 📞 Support

For issues or questions:
- Check logs: `tail -f logs/api.log`
- Test locally first: `curl http://localhost:7000/health`
- Restart server: `pkill -f api_server && ./start_api.sh`

---

**Ready to deploy!** 🚀
