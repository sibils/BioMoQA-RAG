#!/bin/bash
# Start BioMoQA RAG FastAPI Server

cd /home/egaillac/BioMoQA-RAG

echo "Starting BioMoQA RAG API Server..."
echo "Server will be available at: http://egaillac.lan.text-analytics.ch:9000"
echo ""

# Start with python directly
./venv/bin/python3 -m uvicorn api_server:app --host 0.0.0.0 --port 9000
