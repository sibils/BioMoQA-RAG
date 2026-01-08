#!/usr/bin/env python3
"""
Clean V2 results to keep only questions 1-120, removing duplicates.
"""

import pandas as pd

# Load with proper CSV parsing
df = pd.read_csv("results/biomoqa_120_v2_results.csv")

print(f"Total rows loaded: {len(df)}")
print(f"Unique question IDs: {df['question_id'].nunique()}")
print(f"Question ID range: {df['question_id'].min()} - {df['question_id'].max()}")

# Filter to questions 1-120 only
df_120 = df[df['question_id'] <= 120].copy()
print(f"\nAfter filtering to 1-120: {len(df_120)} rows")

# Remove duplicates, keeping first occurrence
df_clean = df_120.drop_duplicates(subset=['question_id'], keep='first')
print(f"After removing duplicates: {len(df_clean)} rows")

# Sort by question_id
df_clean = df_clean.sort_values('question_id').reset_index(drop=True)

# Check for missing questions
all_ids = set(range(1, 121))
present_ids = set(df_clean['question_id'])
missing_ids = sorted(all_ids - present_ids)

if missing_ids:
    print(f"\n⚠ Missing {len(missing_ids)} questions: {missing_ids[:10]}...")
else:
    print(f"\n✓ All 120 questions present")

# Save clean version
output_path = "results/biomoqa_120_v2_results.csv"
df_clean.to_csv(output_path, index=False)
print(f"\n✓ Saved clean results to {output_path}")
print(f"  Total: {len(df_clean)} questions")

# Quick stats
print(f"\n" + "="*60)
print("V2 Evaluation Summary")
print("="*60)
print(f"Questions processed: {len(df_clean)}")
print(f"Avg pipeline time: {df_clean['pipeline_time'].mean():.2f}s")
print(f"Avg documents retrieved: {df_clean['num_retrieved'].mean():.1f}")
print(f"Total evaluation time: {df_clean['pipeline_time'].sum()/60:.1f} minutes")
