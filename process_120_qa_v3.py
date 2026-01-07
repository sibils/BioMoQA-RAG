#!/usr/bin/env python3
"""
Process 120 QA pairs through V3 pipeline and save results.
"""

import pandas as pd
import requests
import time
from tqdm import tqdm
from pathlib import Path

# Will use API server for V3
API_URL = "http://localhost:9000"

def main():
    # Load dataset
    csv_path = "/home/egaillac/Biomoqa/data/Question generation - biotXplorer - June 2024.csv"
    print(f"Loading dataset from {csv_path}...")

    df = pd.read_csv(csv_path)
    df = df.iloc[2:, 4:].reset_index(drop=True)
    df.columns = ["question", "golden_answer", "gold_context"]

    print(f"Loaded {len(df)} QA pairs")

    # Check API is running
    print("\nChecking V3 API connection...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        health = response.json()
        print(f"✓ API connected: {health}")
        if health.get('pipeline_version') != 'v3':
            print(f"⚠ Warning: API is running {health.get('pipeline_version')}, expected v3")
    except Exception as e:
        print(f"✗ API not running! Please start V3 API first:")
        print(f"  cd /home/egaillac/BioMoQA-RAG")
        print(f"  ./venv/bin/python3 -m uvicorn api_server_v3:app --host 0.0.0.0 --port 9000")
        return

    # Process via V3 API
    print(f"\nProcessing {len(df)} questions through V3...")
    results = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            response = requests.post(
                f"{API_URL}/qa",
                json={"question": row["question"], "debug": True},
                timeout=60
            )

            if response.status_code != 200:
                print(f"\n✗ Question {idx+1} failed: HTTP {response.status_code}")
                errors.append({"question_id": idx + 1, "error": f"HTTP {response.status_code}"})
                continue

            result = response.json()

            results.append({
                "question_id": idx + 1,
                "question": row["question"],
                "golden_answer": row["golden_answer"],
                "model_answer": result["answer"],
                "gold_context": row["gold_context"],
                "pipeline_time": result["pipeline_time"],
                "pipeline_version": "v3",
                "num_retrieved": result.get("num_retrieved", 0),
                "debug_info": str(result.get("debug_info", {}))
            })

            # Rate limiting
            time.sleep(0.5)

        except requests.Timeout:
            print(f"\n✗ Question {idx+1} timed out")
            errors.append({"question_id": idx + 1, "error": "Timeout"})
        except Exception as e:
            print(f"\n✗ Question {idx+1} error: {e}")
            errors.append({"question_id": idx + 1, "error": str(e)})

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "biomoqa_120_v3_results.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(results)} results to {output_path}")

    if errors:
        error_path = output_dir / "biomoqa_120_v3_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"✗ {len(errors)} errors saved to {error_path}")

    # Quick stats
    if results:
        df_results = pd.DataFrame(results)
        avg_time = df_results["pipeline_time"].mean()
        avg_docs = df_results["num_retrieved"].mean()

        print(f"\n{'='*60}")
        print(f"V3 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total processed: {len(results)}/{len(df)}")
        print(f"Success rate: {len(results)/len(df)*100:.1f}%")
        print(f"Avg time per question: {avg_time:.2f}s")
        print(f"Avg documents retrieved: {avg_docs:.1f}")
        print(f"\nNext step: Compare with V2")
        print(f"  ./venv/bin/python3 compare_v2_v3.py")

if __name__ == "__main__":
    main()
