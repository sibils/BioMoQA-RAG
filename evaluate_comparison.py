"""
Comparison: Old-style SIBILS BioBERT QA vs New extractive RAG pipeline.

Old system (mimic of biodiversitypmc.sibils.org/api/QA):
  - Same SIBILS BM25 API, medline collection, n=5 docs (their default)
  - No reranking, no relevance filter
  - True single-pass BioBERT: one forward pass per doc, doc_stride=0, max_seq_len=512

New system:
  - BM25 + FAISS hybrid retrieval, all collections
  - CrossEncoder reranking + relevance filter
  - Two-pass BioBERT with sliding window

Benchmark: BioASQ (kroshan/BioASQ on HuggingFace) — deduplicated factoid questions.

Usage:
    python evaluate_comparison.py --limit 50
    python evaluate_comparison.py --limit 50 --output results/comparison.csv
"""

import argparse
import csv
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def squad_f1(prediction: str, ground_truth: str) -> float:
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
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def is_answered(text: str) -> bool:
    return bool(text and text.strip() and len(text.strip()) > 2)


# ---------------------------------------------------------------------------
# Old system mimic: same SIBILS BM25 API, medline only, no reranker,
# no relevance filter, true single-pass BioBERT (direct qa() call, 512 tokens max).
# This matches what biodiversitypmc.sibils.org/api/QA does internally.
# ---------------------------------------------------------------------------

_old_retriever = None
_old_qa = None


def get_old_components():
    global _old_retriever, _old_qa
    if _old_retriever is None:
        from src.biomoqa_rag.retrieval.sibils_retriever import SIBILSRetriever
        from transformers import pipeline as hf_pipeline
        print("  Loading old-mimic components (SIBILS BM25 medline + single-pass BioBERT)…")
        _old_retriever = SIBILSRetriever(collection="medline")
        # Same model as old system: BioBERT fine-tuned on SQuAD2
        _old_qa = hf_pipeline(
            "question-answering",
            model="ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
            handle_impossible_answer=False,
            device=-1,
        )
        print("  Old-mimic components ready.")
    return _old_retriever, _old_qa


def run_old_system(question: str, n: int = 5) -> tuple[str, float]:
    """
    Mimic old sibils.org QA:
    - SIBILS BM25 search (medline, n=5 — their default)
    - No reranking, no relevance filter
    - True single-pass BioBERT: one forward pass per doc, first 512 tokens only
    """
    retriever, qa = get_old_components()
    t0 = time.time()
    try:
        docs = retriever.retrieve(question, n=n)
        if not docs:
            return "", time.time() - t0
        # Build context strings: title + abstract, truncated to ~2000 chars
        # (old system passed full abstract to BioBERT single window)
        contexts = [
            ((doc.title.strip() + ". " if doc.title and doc.title.strip() else "") + (doc.abstract or ""))[:2000]
            for doc in docs
        ]
        contexts = [c for c in contexts if c.strip()]
        if not contexts:
            return "", time.time() - t0
        # Single forward pass — no sliding window (old system default)
        inputs = [{"question": question, "context": ctx} for ctx in contexts]
        results = qa(inputs, batch_size=len(inputs), max_seq_len=512, doc_stride=0)
        if isinstance(results, dict):
            results = [results]
        # Pick highest-scoring answer
        best = max(results, key=lambda r: r["score"])
        answer = best["answer"].strip()
        return answer, time.time() - t0
    except Exception as e:
        print(f"    [old mimic error] {e}")
        return "", time.time() - t0


# ---------------------------------------------------------------------------
# New system: full RAGPipeline, extractive mode
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.biomoqa_rag.pipeline import RAGPipeline, RAGConfig
        print("  Loading new pipeline (extractive, no LLM)…")
        config = RAGConfig(use_vllm=False, use_cpu=False)
        _pipeline = RAGPipeline(config)
        print("  New pipeline ready.")
    return _pipeline


def run_new_system(question: str) -> tuple[str, float]:
    """Run new extractive pipeline; returns (best_answer_text, elapsed_seconds)."""
    t0 = time.time()
    try:
        result = get_pipeline().run(question=question, mode="extractive", return_documents=False)
        answers = result.get("answers", []) or []
        best = answers[0].get("answer", "").strip() if answers else ""
        return best, time.time() - t0
    except Exception as e:
        print(f"    [new pipeline error] {e}")
        return "", time.time() - t0


# ---------------------------------------------------------------------------
# BioASQ loader
# ---------------------------------------------------------------------------

def load_bioasq(limit: int | None = None) -> list[dict]:
    print("  Downloading BioASQ from HuggingFace…")
    from datasets import load_dataset
    ds = load_dataset("kroshan/BioASQ", split="train")
    seen = set()
    rows = []
    for item in ds:
        q = item.get("question", "").strip()
        text = item.get("text", "")
        m = re.search(r"<answer>\s*(.+?)\s*<context>", text, re.DOTALL)
        gold = m.group(1).strip() if m else ""
        if q and gold and q not in seen:
            seen.add(q)
            rows.append({"question": q, "golden_answer": gold})
        if limit and len(rows) >= limit:
            break
    print(f"  Loaded {len(rows)} unique BioASQ questions")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of BioASQ questions to evaluate (default: 50)")
    parser.add_argument("--output", default="results/comparison_old_vs_new.csv")
    parser.add_argument("--questions", nargs="*",
                        help="Custom questions (no golden answer — shows answers only)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    if args.questions:
        rows = [{"question": q, "golden_answer": ""} for q in args.questions]
        has_gold = False
    else:
        rows = load_bioasq(limit=args.limit)
        has_gold = True

    print(f"\nEvaluating {len(rows)} questions — old vs new (extractive)")
    print("=" * 70)

    fieldnames = [
        "question_id", "question", "golden_answer",
        "old_answer", "new_answer",
        "old_answered", "new_answered",
        "old_f1", "new_f1",
        "old_rouge1", "new_rouge1",
        "old_rougeL", "new_rougeL",
        "old_time", "new_time",
    ]

    totals = {
        "old": {"answered": 0, "f1": 0.0, "rouge1": 0.0, "rougeL": 0.0, "time": 0.0},
        "new": {"answered": 0, "f1": 0.0, "rouge1": 0.0, "rougeL": 0.0, "time": 0.0},
    }

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, row in enumerate(rows):
            qid = i + 1
            question = row["question"]
            golden = row["golden_answer"]

            print(f"\n[{qid}/{len(rows)}] {question[:80]}")
            if golden:
                print(f"  gold: {golden[:60]}")

            old_answer, old_time = run_old_system(question)
            new_answer, new_time = run_new_system(question)

            old_answered = is_answered(old_answer)
            new_answered = is_answered(new_answer)

            print(f"  old ({old_time:.1f}s): {old_answer[:70] if old_answered else '(no answer)'}")
            print(f"  new ({new_time:.1f}s): {new_answer[:70] if new_answered else '(no answer)'}")

            if has_gold and golden:
                old_f1 = squad_f1(old_answer, golden) if old_answered else 0.0
                new_f1 = squad_f1(new_answer, golden) if new_answered else 0.0
                old_rouge = compute_rouge(old_answer, golden) if old_answered else {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
                new_rouge = compute_rouge(new_answer, golden) if new_answered else {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
                print(f"  old F1={old_f1:.3f}  new F1={new_f1:.3f}")
            else:
                old_f1 = new_f1 = 0.0
                old_rouge = new_rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

            totals["old"]["answered"] += int(old_answered)
            totals["new"]["answered"] += int(new_answered)
            totals["old"]["f1"] += old_f1
            totals["new"]["f1"] += new_f1
            totals["old"]["rouge1"] += old_rouge["rouge1"]
            totals["new"]["rouge1"] += new_rouge["rouge1"]
            totals["old"]["rougeL"] += old_rouge["rougeL"]
            totals["new"]["rougeL"] += new_rouge["rougeL"]
            totals["old"]["time"] += old_time
            totals["new"]["time"] += new_time

            writer.writerow({
                "question_id": qid,
                "question": question,
                "golden_answer": golden,
                "old_answer": old_answer,
                "new_answer": new_answer,
                "old_answered": old_answered,
                "new_answered": new_answered,
                "old_f1": round(old_f1, 4),
                "new_f1": round(new_f1, 4),
                "old_rouge1": round(old_rouge["rouge1"], 4),
                "new_rouge1": round(new_rouge["rouge1"], 4),
                "old_rougeL": round(old_rouge["rougeL"], 4),
                "new_rougeL": round(new_rouge["rougeL"], 4),
                "old_time": round(old_time, 2),
                "new_time": round(new_time, 2),
            })
            f.flush()

    n = len(rows)
    print("\n" + "=" * 70)
    print(f"{'Metric':<20} {'Old (SIBILS BioBERT)':>22} {'New (RAG extractive)':>22}")
    print("-" * 66)
    print(f"{'Answer rate':<20} {totals['old']['answered']/n*100:>21.1f}% {totals['new']['answered']/n*100:>21.1f}%")
    if has_gold:
        print(f"{'F1 (avg)':<20} {totals['old']['f1']/n:>22.4f} {totals['new']['f1']/n:>22.4f}")
        print(f"{'ROUGE-1 (avg)':<20} {totals['old']['rouge1']/n:>22.4f} {totals['new']['rouge1']/n:>22.4f}")
        print(f"{'ROUGE-L (avg)':<20} {totals['old']['rougeL']/n:>22.4f} {totals['new']['rougeL']/n:>22.4f}")
    print(f"{'Avg time/q (s)':<20} {totals['old']['time']/n:>22.1f} {totals['new']['time']/n:>22.1f}")
    print(f"\nResults saved to: {output_path}")

    summary = {
        "n_questions": n,
        "old": {k: round(v / n, 4) if k != "answered" else round(v / n * 100, 1) for k, v in totals["old"].items()},
        "new": {k: round(v / n, 4) if k != "answered" else round(v / n * 100, 1) for k, v in totals["new"].items()},
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to:  {summary_path}")


if __name__ == "__main__":
    main()
