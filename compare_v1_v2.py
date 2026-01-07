#!/usr/bin/env python3
"""
Compare V1 vs V2 evaluation results.
"""

import pandas as pd
from pathlib import Path
import json

def calculate_metrics(results_path):
    """Calculate evaluation metrics from results CSV."""
    from rouge import Rouge

    df = pd.read_csv(results_path)

    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = []

    for _, row in df.iterrows():
        try:
            scores = rouge.get_scores(
                row['model_answer'],
                row['golden_answer'],
                avg=True
            )
            rouge_scores.append(scores)
        except:
            continue

    # Average ROUGE scores
    avg_rouge = {
        'rouge-1': {
            'f': sum(s['rouge-1']['f'] for s in rouge_scores) / len(rouge_scores),
            'p': sum(s['rouge-1']['p'] for s in rouge_scores) / len(rouge_scores),
            'r': sum(s['rouge-1']['r'] for s in rouge_scores) / len(rouge_scores),
        },
        'rouge-2': {
            'f': sum(s['rouge-2']['f'] for s in rouge_scores) / len(rouge_scores),
            'p': sum(s['rouge-2']['p'] for s in rouge_scores) / len(rouge_scores),
            'r': sum(s['rouge-2']['r'] for s in rouge_scores) / len(rouge_scores),
        },
        'rouge-l': {
            'f': sum(s['rouge-l']['f'] for s in rouge_scores) / len(rouge_scores),
            'p': sum(s['rouge-l']['p'] for s in rouge_scores) / len(rouge_scores),
            'r': sum(s['rouge-l']['r'] for s in rouge_scores) / len(rouge_scores),
        }
    }

    # Calculate other metrics
    avg_time = df['pipeline_time'].mean()
    avg_answer_len = df['model_answer'].str.len().mean()

    return {
        'rouge': avg_rouge,
        'avg_time': avg_time,
        'avg_answer_len': avg_answer_len,
        'total_questions': len(df)
    }

def main():
    v1_path = Path("results/biomoqa_120_results.csv")
    v2_path = Path("results/biomoqa_120_v2_results.csv")

    if not v1_path.exists():
        print(f"✗ V1 results not found: {v1_path}")
        return

    if not v2_path.exists():
        print(f"✗ V2 results not found: {v2_path}")
        print("  Run: ./venv/bin/python3 process_120_qa_v2.py")
        return

    print("="*80)
    print("V1 vs V2 Performance Comparison")
    print("="*80)
    print()

    # Load results
    v1_metrics = calculate_metrics(v1_path)
    v2_metrics = calculate_metrics(v2_path)

    # Print comparison
    print("ROUGE-1 Scores (F1):")
    print(f"  V1: {v1_metrics['rouge']['rouge-1']['f']*100:.2f}%")
    print(f"  V2: {v2_metrics['rouge']['rouge-1']['f']*100:.2f}%")
    improvement = (v2_metrics['rouge']['rouge-1']['f'] - v1_metrics['rouge']['rouge-1']['f']) / v1_metrics['rouge']['rouge-1']['f'] * 100
    print(f"  Improvement: {improvement:+.1f}%")
    print()

    print("ROUGE-2 Scores (F1):")
    print(f"  V1: {v1_metrics['rouge']['rouge-2']['f']*100:.2f}%")
    print(f"  V2: {v2_metrics['rouge']['rouge-2']['f']*100:.2f}%")
    improvement = (v2_metrics['rouge']['rouge-2']['f'] - v1_metrics['rouge']['rouge-2']['f']) / v1_metrics['rouge']['rouge-2']['f'] * 100
    print(f"  Improvement: {improvement:+.1f}%")
    print()

    print("ROUGE-L Scores (F1):")
    print(f"  V1: {v1_metrics['rouge']['rouge-l']['f']*100:.2f}%")
    print(f"  V2: {v2_metrics['rouge']['rouge-l']['f']*100:.2f}%")
    improvement = (v2_metrics['rouge']['rouge-l']['f'] - v1_metrics['rouge']['rouge-l']['f']) / v1_metrics['rouge']['rouge-l']['f'] * 100
    print(f"  Improvement: {improvement:+.1f}%")
    print()

    print("Pipeline Performance:")
    print(f"  V1 avg time: {v1_metrics['avg_time']:.2f}s")
    print(f"  V2 avg time: {v2_metrics['avg_time']:.2f}s")
    time_diff = v2_metrics['avg_time'] - v1_metrics['avg_time']
    print(f"  Difference: {time_diff:+.2f}s ({time_diff/v1_metrics['avg_time']*100:+.1f}%)")
    print()

    print("Answer Quality:")
    print(f"  V1 avg answer length: {v1_metrics['avg_answer_len']:.0f} chars")
    print(f"  V2 avg answer length: {v2_metrics['avg_answer_len']:.0f} chars")
    print()

    # Save detailed comparison
    comparison = {
        'v1': v1_metrics,
        'v2': v2_metrics,
        'improvements': {
            'rouge-1': (v2_metrics['rouge']['rouge-1']['f'] - v1_metrics['rouge']['rouge-1']['f']) / v1_metrics['rouge']['rouge-1']['f'] * 100,
            'rouge-2': (v2_metrics['rouge']['rouge-2']['f'] - v1_metrics['rouge']['rouge-2']['f']) / v1_metrics['rouge']['rouge-2']['f'] * 100,
            'rouge-l': (v2_metrics['rouge']['rouge-l']['f'] - v1_metrics['rouge']['rouge-l']['f']) / v1_metrics['rouge']['rouge-l']['f'] * 100,
            'time': time_diff
        }
    }

    output_path = Path("results/v1_v2_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"✓ Detailed comparison saved to {output_path}")
    print()

    # Verdict
    rouge1_improvement = comparison['improvements']['rouge-1']
    print("="*80)
    print("VERDICT:")
    print("="*80)

    if rouge1_improvement > 30:
        print("🎉 EXCELLENT improvement! V2 significantly outperforms V1.")
    elif rouge1_improvement > 15:
        print("✓ GOOD improvement! V2 shows meaningful gains over V1.")
    elif rouge1_improvement > 5:
        print("✓ MODERATE improvement. Consider additional enhancements.")
    else:
        print("⚠ LIMITED improvement. Review V2 components and consider alternatives.")

    print()
    print("Next steps:")
    if rouge1_improvement > 20:
        print("  → Deploy V2 as default")
        print("  → Consider Phase 2: Hybrid retrieval (BM25 + dense)")
        print("  → Consider Phase 3: Medical-domain reranker")
    else:
        print("  → Analyze debug logs to understand bottlenecks")
        print("  → Test individual V2 components")
        print("  → Consider different reranker models")

if __name__ == "__main__":
    main()
