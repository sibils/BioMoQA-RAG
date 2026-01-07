#!/usr/bin/env python3
"""
Evaluate BioMoQA RAG Results

Generates comprehensive evaluation metrics:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU scores
- Exact Match (EM)
- F1 scores
- Average response times
- Citation statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re


def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, ground_truth):
    """Compute exact match score."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_rouge_scores(prediction, ground_truth):
    """Compute ROUGE scores (simple implementation)."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    # ROUGE-1: unigram overlap
    pred_unigrams = set(pred_tokens)
    truth_unigrams = set(truth_tokens)
    overlap = len(pred_unigrams & truth_unigrams)
    rouge1 = overlap / max(len(truth_unigrams), 1)

    # ROUGE-2: bigram overlap
    pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
    truth_bigrams = set(zip(truth_tokens[:-1], truth_tokens[1:]))
    overlap = len(pred_bigrams & truth_bigrams)
    rouge2 = overlap / max(len(truth_bigrams), 1)

    # ROUGE-L: longest common subsequence
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs = lcs_length(pred_tokens, truth_tokens)
    rougeL = lcs / max(len(truth_tokens), 1)

    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}


def evaluate_results(csv_path: str):
    """Evaluate all results and generate comprehensive report."""

    print("="*80)
    print("BioMoQA RAG Evaluation Report")
    print("="*80)
    print()

    # Load results
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} QA pairs\n")

    # Filter out errors
    df_valid = df[~df['model_answer'].str.startswith('ERROR:', na=False)]
    num_errors = len(df) - len(df_valid)

    if num_errors > 0:
        print(f"⚠ Warning: {num_errors} questions had errors and were excluded")
        print()

    # Compute metrics for each row
    results = []
    for idx, row in df_valid.iterrows():
        pred = str(row['model_answer'])
        gold = str(row['golden_answer'])

        em = compute_exact_match(pred, gold)
        f1 = compute_f1(pred, gold)
        rouge = compute_rouge_scores(pred, gold)

        results.append({
            'question_id': row['question_id'],
            'exact_match': em,
            'f1': f1,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'pipeline_time': row['pipeline_time_seconds'],
            'num_citations': len(str(row['citations']).split(',')) if pd.notna(row['citations']) and str(row['citations']).strip() else 0
        })

    results_df = pd.DataFrame(results)

    # Aggregate statistics
    print("OVERALL METRICS")
    print("-" * 80)
    print(f"Exact Match (EM):        {results_df['exact_match'].mean():.2%} ({results_df['exact_match'].sum()}/{len(results_df)} questions)")
    print(f"Average F1 Score:        {results_df['f1'].mean():.2%}")
    print(f"Average ROUGE-1:         {results_df['rouge1'].mean():.2%}")
    print(f"Average ROUGE-2:         {results_df['rouge2'].mean():.2%}")
    print(f"Average ROUGE-L:         {results_df['rougeL'].mean():.2%}")
    print()

    print("PERFORMANCE METRICS")
    print("-" * 80)
    print(f"Average time/question:   {results_df['pipeline_time'].mean():.2f}s")
    print(f"Median time/question:    {results_df['pipeline_time'].median():.2f}s")
    print(f"Min time:                {results_df['pipeline_time'].min():.2f}s")
    print(f"Max time:                {results_df['pipeline_time'].max():.2f}s")
    print(f"Total processing time:   {results_df['pipeline_time'].sum():.0f}s ({results_df['pipeline_time'].sum()/60:.1f} minutes)")
    print()

    print("CITATION STATISTICS")
    print("-" * 80)
    print(f"Average citations/answer: {results_df['num_citations'].mean():.1f}")
    print(f"Questions with citations: {(results_df['num_citations'] > 0).sum()}/{len(results_df)} ({(results_df['num_citations'] > 0).mean():.1%})")
    print(f"Max citations:            {results_df['num_citations'].max():.0f}")
    print()

    print("RESPONSE LENGTH STATISTICS")
    print("-" * 80)
    avg_response_length = df_valid['response_length_chars'].mean()
    print(f"Average response length:  {avg_response_length:.0f} characters")
    print(f"Median response length:   {df_valid['response_length_chars'].median():.0f} characters")
    print(f"Min response length:      {df_valid['response_length_chars'].min():.0f} characters")
    print(f"Max response length:      {df_valid['response_length_chars'].max():.0f} characters")
    print()

    # F1 Score distribution
    print("F1 SCORE DISTRIBUTION")
    print("-" * 80)
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    f1_dist = pd.cut(results_df['f1'], bins=bins, labels=labels, include_lowest=True)
    for label in labels:
        count = (f1_dist == label).sum()
        pct = count / len(results_df)
        print(f"{label:>10}: {count:3d} questions ({pct:5.1%}) {'█' * int(pct * 50)}")
    print()

    # Save detailed results
    output_path = Path(csv_path).parent / "evaluation_detailed.csv"

    # Merge results back with original data
    eval_df = df_valid.merge(results_df[['question_id', 'exact_match', 'f1', 'rouge1', 'rouge2', 'rougeL']],
                              on='question_id', how='left')
    eval_df.to_csv(output_path, index=False)

    print(f"✓ Detailed evaluation saved to: {output_path}")
    print()

    # Generate summary statistics file
    summary_path = Path(csv_path).parent / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("BioMoQA RAG Evaluation Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total questions: {len(df_valid)}\n")
        f.write(f"Exact Match: {results_df['exact_match'].mean():.2%}\n")
        f.write(f"Average F1: {results_df['f1'].mean():.2%}\n")
        f.write(f"Average ROUGE-1: {results_df['rouge1'].mean():.2%}\n")
        f.write(f"Average ROUGE-2: {results_df['rouge2'].mean():.2%}\n")
        f.write(f"Average ROUGE-L: {results_df['rougeL'].mean():.2%}\n")
        f.write(f"Average time: {results_df['pipeline_time'].mean():.2f}s\n")

    print(f"✓ Summary saved to: {summary_path}")
    print()

    # Show some examples
    print("EXAMPLE COMPARISONS (Top 3 F1 Scores)")
    print("="*80)
    top_3 = eval_df.nlargest(3, 'f1')
    for idx, row in top_3.iterrows():
        print(f"\nQuestion {row['question_id']}: {row['question']}")
        print(f"F1 Score: {row['f1']:.2%}")
        print(f"Golden Answer: {row['golden_answer'][:100]}...")
        print(f"Model Answer:  {row['model_answer'][:100]}...")
        print("-"*80)

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

    return results_df


if __name__ == "__main__":
    csv_path = "results/biomoqa_120_results.csv"
    evaluate_results(csv_path)
