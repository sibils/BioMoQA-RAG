#!/bin/bash
# Test V2 improvements with correct Python environment

cd /home/egaillac/BioMoQA-RAG

# Make sure GPU is free
echo "Checking GPU..."
nvidia-smi | grep VLLM | awk '{print $5}' | xargs -r kill -9 2>/dev/null
sleep 2

echo "Starting V2 test with venv Python..."
echo ""

# Use venv python
./venv/bin/python3 test_v2_improvements.py
