#!/usr/bin/env python3
"""
Batch Process 120 QA Pairs from BioMoQA Dataset

Generates CSV with:
- Question
- Golden Answer
- Gold Context
- Model Answer
- Top Retrieved Context
- Pipeline Time
- Citations
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import time

from src.pipeline_vllm import FastRAGPipeline, RAGConfig


def load_biomoqa_dataset(csv_path: str) -> pd.DataFrame:
    """Load the 120 QA pairs from biotXplorer CSV."""
    # Read CSV (skip first 2 rows as per old code)
    df1 = pd.read_csv(csv_path)
    df2 = df1.iloc[2:, 4:]  # Extract Question, Answer, Context columns
    df = df2.reset_index(drop=True)

    # Rename columns
    df.columns = ["question", "golden_answer", "gold_context"]

    # Remove any empty rows
    df = df.dropna(subset=["question"])

    print(f"Loaded {len(df)} QA pairs from {csv_path}")
    return df


def process_batch(
    df: pd.DataFrame,
    pipeline: FastRAGPipeline,
    output_dir: str = "results",
):
    """Process all questions and generate CSV."""

    results = []

    print(f"\nProcessing {len(df)} questions...")
    print("="*80)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        question = row["question"]
        golden_answer = row["golden_answer"]
        gold_context = row["gold_context"]

        try:
            # Run RAG pipeline
            result = pipeline.run(
                question=question,
                topic_id=f"Q{idx+1:03d}",
                return_documents=True,
            )

            # Extract top retrieved context
            top_contexts = []
            if "documents" in result:
                for doc in result["documents"][:5]:  # Top 5
                    top_contexts.append(
                        f"[{doc['id']}] {doc['title']}: {doc['abstract']}"
                    )
            top_retrieved_context = "\n\n".join(top_contexts)

            # Extract model answer (combine all sentences)
            model_answer_sentences = [
                sent["text"] for sent in result["answer"]
            ]
            model_answer = " ".join(model_answer_sentences)

            # Extract citations
            all_citations = []
            for sent in result["answer"]:
                all_citations.extend(sent["citations"])
            citations_str = ", ".join([str(c) for c in set(all_citations)])

            # Build result row
            results.append({
                "question_id": idx + 1,
                "question": question,
                "golden_answer": golden_answer,
                "model_answer": model_answer,
                "gold_context": gold_context,
                "top_retrieved_context": top_retrieved_context,
                "citations": citations_str,
                "pipeline_time_seconds": result["pipeline_time"],
                "num_documents_retrieved": result["num_retrieved"],
                "response_length_chars": result["response_length"],
            })

        except Exception as e:
            print(f"\nError processing question {idx+1}: {str(e)}")
            results.append({
                "question_id": idx + 1,
                "question": question,
                "golden_answer": golden_answer,
                "model_answer": f"ERROR: {str(e)}",
                "gold_context": gold_context,
                "top_retrieved_context": "",
                "citations": "",
                "pipeline_time_seconds": 0,
                "num_documents_retrieved": 0,
                "response_length_chars": 0,
            })

        # Small delay to avoid overwhelming API
        time.sleep(0.1)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path(output_dir) / "biomoqa_120_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Results saved to: {output_path}")

    # Also save as JSON for detailed analysis
    json_path = Path(output_dir) / "biomoqa_120_results.json"
    results_df.to_json(json_path, orient="records", indent=2)
    print(f"✓ JSON saved to: {json_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total questions processed: {len(results_df)}")
    print(f"Average pipeline time: {results_df['pipeline_time_seconds'].mean():.2f}s")
    print(f"Median pipeline time: {results_df['pipeline_time_seconds'].median():.2f}s")
    print(f"Average response length: {results_df['response_length_chars'].mean():.0f} chars")
    print(f"Total processing time: {results_df['pipeline_time_seconds'].sum():.0f}s ({results_df['pipeline_time_seconds'].sum()/60:.1f} minutes)")

    return results_df


def main():
    """Main execution."""
    print("="*80)
    print("BioMoQA-Ragnarök: Batch Processing 120 QA Pairs")
    print("="*80)
    print()

    # Paths
    dataset_path = "/home/egaillac/Biomoqa/data/Question generation - biotXplorer - June 2024.csv"

    # Load dataset
    df = load_biomoqa_dataset(dataset_path)

    print(f"\nFirst 3 questions:")
    print("-"*80)
    for i in range(min(3, len(df))):
        print(f"{i+1}. {df.iloc[i]['question']}")
    print()

    # Initialize pipeline
    print("Initializing Fast RAG Pipeline (vLLM)...")
    config = RAGConfig(
        retrieval_n=50,
        rerank_n=20,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_vllm=True,
        gpu_memory_utilization=0.7,
    )

    pipeline = FastRAGPipeline(config)

    # Process all questions
    results_df = process_batch(df, pipeline)

    print("\n" + "="*80)
    print("✓ Batch processing complete!")
    print("="*80)
    print("\nOutput files:")
    print("  - results/biomoqa_120_results.csv")
    print("  - results/biomoqa_120_results.json")
    print("\nYou can now:")
    print("  1. Open the CSV in Excel/LibreOffice")
    print("  2. Compare model_answer vs golden_answer")
    print("  3. Run evaluation metrics (ROUGE, BERTScore)")


if __name__ == "__main__":
    main()
