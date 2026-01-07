#!/bin/bash
# Monitor V2 evaluation progress

echo "V2 Evaluation Monitor"
echo "===================="
echo

# Check if process is running
PID=$(ps aux | grep "process_120_qa_v2.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "✗ Evaluation process not running"

    # Check if results exist
    if [ -f "results/biomoqa_120_v2_results.csv" ]; then
        LINES=$(wc -l < results/biomoqa_120_v2_results.csv)
        echo "✓ Results file exists with $LINES lines"
    else
        echo "✗ No results file found"
    fi
    exit 0
fi

echo "✓ Process running (PID: $PID)"
echo

# Check results file
if [ -f "results/biomoqa_120_v2_results.csv" ]; then
    COMPLETED=$(($(wc -l < results/biomoqa_120_v2_results.csv) - 1))
    PROGRESS=$((COMPLETED * 100 / 120))
    echo "Progress: $COMPLETED/120 questions ($PROGRESS%)"

    # Estimate time remaining
    if [ $COMPLETED -gt 0 ]; then
        # Check last 5 questions timing
        if [ -f "results/biomoqa_120_v2_results.csv" ]; then
            AVG_TIME=$(tail -5 results/biomoqa_120_v2_results.csv | awk -F',' '{sum+=$6; count++} END {print sum/count}')
            REMAINING=$((120 - COMPLETED))
            EST_SECONDS=$(echo "$AVG_TIME * $REMAINING" | bc)
            EST_MINUTES=$(echo "scale=1; $EST_SECONDS / 60" | bc)
            echo "Estimated time remaining: ${EST_MINUTES} minutes"
        fi
    fi
else
    echo "Waiting for results file to be created..."
fi

echo
echo "To follow live progress:"
echo "  watch -n 5 ./monitor_evaluation.sh"
