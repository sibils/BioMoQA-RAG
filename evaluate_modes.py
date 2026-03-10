"""
Comparison evaluation: generative (LLM) vs extractive (BioBERT) mode.

Runs both pipeline modes on the 120 benchmark questions and computes
F1, ROUGE-1/2/L, answer rate, and speed. Outputs per-question CSV
and a summary table.

Usage:
    python evaluate_modes.py                        # run both modes
    python evaluate_modes.py --modes extractive     # extractive only
    python evaluate_modes.py --modes generative     # generative only
    python evaluate_modes.py --limit 10             # first N questions only
    python evaluate_modes.py --resume               # skip already-done rows
"""

import argparse
import csv
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def squad_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 as used in SQuAD evaluation."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge(prediction: str, ground_truth: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


NO_ANSWER_MARKER = "no relevant answer found"


def is_answered(text: str) -> bool:
    return NO_ANSWER_MARKER not in text.lower()


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

def load_pipeline(modes: list):
    """Load the RAG pipeline. Skips LLM loading when only running extractive mode."""
    from src.biomoqa_rag.pipeline import RAGPipeline, RAGConfig

    if modes == ["extractive"]:
        # No LLM needed — disable both vLLM and CPU inference so only
        # retrieval + BioBERT extractive QA are loaded.
        config = RAGConfig(use_vllm=False, use_cpu=False)
    else:
        config = RAGConfig()
    return RAGPipeline(config)


def run_query(pipeline, question: str, mode: str) -> tuple[str, float]:
    """Run a question through the pipeline; returns (answer_text, elapsed_seconds)."""
    t0 = time.time()
    result = pipeline.run(question=question, mode=mode, return_documents=False)
    elapsed = time.time() - t0
    answer_text = " ".join(s["text"] for s in result["answer"]).strip()
    return answer_text, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["generative", "extractive"],
                        choices=["generative", "extractive"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions")
    parser.add_argument("--resume", action="store_true",
                        help="Skip question IDs already present in output CSV")
    parser.add_argument("--input", default="results/biomoqa_120_results.csv",
                        help="Input CSV with question/golden_answer columns")
    parser.add_argument("--output", default="results/mode_comparison.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load benchmark questions
    df = pd.read_csv(input_path)
    df = df[["question_id", "question", "golden_answer"]].dropna()
    if args.limit:
        df = df.head(args.limit)

    # Resume: skip already-done rows
    done_ids = set()
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        done_ids = set(zip(existing["question_id"], existing["mode"]))
        print(f"Resuming: {len(done_ids)} rows already done")

    print(f"\nEvaluating {len(df)} questions in mode(s): {args.modes}")
    print("=" * 70)

    # Load pipeline once
    pipeline = load_pipeline(args.modes)

    # Open output CSV for appending
    output_path.parent.mkdir(exist_ok=True)
    write_header = not output_path.exists() or not args.resume
    out_file = open(output_path, "a" if args.resume else "w", newline="")
    fieldnames = [
        "question_id", "mode", "question", "golden_answer",
        "model_answer", "is_answered",
        "f1", "rouge1", "rouge2", "rougeL",
        "pipeline_time_seconds",
    ]
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    total = len(df) * len(args.modes)
    done = 0

    for _, row in df.iterrows():
        qid = int(row["question_id"])
        question = str(row["question"])
        golden = str(row["golden_answer"])

        for mode in args.modes:
            if (qid, mode) in done_ids:
                done += 1
                continue

            try:
                answer, elapsed = run_query(pipeline, question, mode)
            except Exception as e:
                print(f"  ERROR q{qid} [{mode}]: {e}")
                answer, elapsed = "", 0.0

            answered = is_answered(answer)
            f1 = squad_f1(answer, golden) if answered else 0.0
            rouge = compute_rouge(answer, golden) if answered else {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

            writer.writerow({
                "question_id": qid,
                "mode": mode,
                "question": question,
                "golden_answer": golden,
                "model_answer": answer,
                "is_answered": int(answered),
                "f1": round(f1, 4),
                "rouge1": round(rouge["rouge1"], 4),
                "rouge2": round(rouge["rouge2"], 4),
                "rougeL": round(rouge["rougeL"], 4),
                "pipeline_time_seconds": round(elapsed, 2),
            })
            out_file.flush()

            done += 1
            print(f"  [{done}/{total}] q{qid} [{mode}] "
                  f"F1={f1:.2%} ROUGE-1={rouge['rouge1']:.2%} "
                  f"answered={'Y' if answered else 'N'} "
                  f"time={elapsed:.1f}s")

    out_file.close()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    results = pd.read_csv(output_path)
    results = results[results["mode"].isin(args.modes)]

    # If only extractive was run, also pull in existing generative metrics for comparison
    existing_gen_path = Path("results/evaluation_detailed.csv")
    if args.modes == ["extractive"] and existing_gen_path.exists():
        gen_df = pd.read_csv(existing_gen_path)
        # Compute answer_rate for generative (non-empty model_answer)
        gen_df["is_answered"] = gen_df["model_answer"].fillna("").apply(
            lambda x: 0 if NO_ANSWER_MARKER in x.lower() else 1
        )
        gen_df["mode"] = "generative (existing)"
        gen_df = gen_df.rename(columns={
            "rouge1": "rouge1",
            "rouge2": "rouge2",
            "rougeL": "rougeL",
            "f1": "f1",
        })
        if args.limit:
            gen_df = gen_df.head(args.limit)
        results = pd.concat([results, gen_df[
            ["mode", "is_answered", "f1", "rouge1", "rouge2", "rougeL"]
        ]], ignore_index=True)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    summary = (
        results.groupby("mode")
        .agg(
            questions=("question_id", "count"),
            answer_rate=("is_answered", "mean"),
            avg_f1=("f1", "mean"),
            avg_rouge1=("rouge1", "mean"),
            avg_rouge2=("rouge2", "mean"),
            avg_rougeL=("rougeL", "mean"),
            avg_time=("pipeline_time_seconds", "mean"),
        )
        .reset_index()
    )

    for col in ["answer_rate", "avg_f1", "avg_rouge1", "avg_rouge2", "avg_rougeL"]:
        summary[col] = summary[col].map(lambda x: f"{x:.2%}")
    summary["avg_time"] = summary["avg_time"].map(lambda x: f"{x:.2f}s")

    print(summary.to_string(index=False))

    # Save summary JSON
    summary_path = output_path.with_name("mode_comparison_summary.json")
    summary.to_json(summary_path, orient="records", indent=2)
    print(f"\nDetailed results  → {output_path}")
    print(f"Summary           → {summary_path}")


if __name__ == "__main__":
    main()
