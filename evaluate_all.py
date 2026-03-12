"""
Cross-dataset evaluation comparing our pipeline vs biodiversitypmc.sibils.org/api/QA.

Datasets:
  1. biomoqa120  – our in-house 120 parasitology/biodiversity questions (gold standard)
  2. bioasq      – BioASQ Task B factoid subset (standard biomedical QA benchmark)
  3. pubmedqa    – PubMedQA pqa_labeled yes/no/maybe subset

For each question we query:
  A. Our pipeline   (extractive mode, no LLM needed)
  B. Old sibils API (biodiversitypmc.sibils.org/api/QA)

Metrics per system: answer_rate, avg_F1, avg_ROUGE-1, avg_ROUGE-L, avg_time
"""

import csv
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, "src")

SIBILS_API = "https://biodiversitypmc.sibils.org/api/QA"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def normalize(text: str) -> list[str]:
    import re, string
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c if c not in string.punctuation else " " for c in text)
    return text.split()


def squad_f1(pred: str, gold: str) -> float:
    p_toks, g_toks = normalize(pred), normalize(gold)
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, gold: str) -> float:
    """Simple token-level ROUGE-L (LCS-based)."""
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def rouge1(pred: str, gold: str) -> float:
    p_toks, g_toks = normalize(pred), normalize(gold)
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


NO_ANSWER_MARKERS = ["cannot be fully answered", "no relevant", "not found", "no answer"]

def is_answered(text: str) -> bool:
    if not text or not text.strip():
        return False
    lo = text.lower()
    return not any(m in lo for m in NO_ANSWER_MARKERS)


# ── dataset loaders ──────────────────────────────────────────────────────────

def load_biomoqa120(limit=None):
    df_path = Path("results/biomoqa_120_results.csv")
    rows = []
    with open(df_path) as f:
        for row in csv.DictReader(f):
            rows.append({"question": row["question"], "golden_answer": row["golden_answer"]})
    if limit:
        rows = rows[:limit]
    return rows


def load_bioasq(limit=50):
    """BioASQ factoid questions — kroshan/BioASQ on HuggingFace.
    text field format: '<answer> ANSWER <context> PASSAGE'
    """
    import re
    print("  Downloading BioASQ …")
    from datasets import load_dataset
    ds = load_dataset("kroshan/BioASQ", split="train")

    rows = []
    for item in ds:
        q = item.get("question", "").strip()
        text = item.get("text", "")
        m = re.search(r"<answer>\s*(.+?)\s*<context>", text, re.DOTALL)
        gold = m.group(1).strip() if m else ""
        if q and gold:
            rows.append({"question": q, "golden_answer": gold})
        if len(rows) >= limit:
            break
    print(f"  Loaded {len(rows)} BioASQ questions")
    return rows


def load_pubmedqa(limit=50):
    """PubMedQA pqa_labeled subset."""
    print("  Downloading PubMedQA …")
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)
    rows = []
    for item in ds:
        q = item.get("question", "")
        gold = item.get("final_decision", "")  # yes / no / maybe
        if q and gold:
            rows.append({"question": q, "golden_answer": gold})
        if len(rows) >= limit:
            break
    print(f"  Loaded {len(rows)} PubMedQA questions")
    return rows


# ── query functions ──────────────────────────────────────────────────────────

def query_sibils(question: str, col: str = "medline", n: int = 5) -> tuple[str, float]:
    t0 = time.time()
    try:
        resp = requests.get(
            SIBILS_API,
            params={"col": col, "q": question, "n": n},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        answers = data.get("answers", [])
        text = answers[0]["answer"] if answers else ""
    except Exception as e:
        text = ""
    return text, time.time() - t0


def query_our_pipeline(pipeline, question: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        result = pipeline.run(question=question, mode="extractive")
        answers = result.get("answers", [])
        text = answers[0]["answer"] if answers else ""
    except Exception as e:
        text = ""
    return text, time.time() - t0


# ── evaluation loop ──────────────────────────────────────────────────────────

FIELDNAMES = [
    "dataset", "question", "golden_answer",
    "our_answer", "our_answered", "our_f1", "our_rouge1", "our_rougeL", "our_time",
    "sib_answer", "sib_answered", "sib_f1", "sib_rouge1", "sib_rougeL", "sib_time",
]


def evaluate_dataset(name: str, questions: list, pipeline, writer, out_file, done_keys: set) -> list[dict]:
    rows = []
    n = len(questions)
    for i, item in enumerate(questions):
        q = item["question"]
        gold = item["golden_answer"]
        key = (name, q[:80])
        if key in done_keys:
            print(f"  [{i+1:3d}/{n}] (skipped — already done)")
            continue

        our_ans, our_time = query_our_pipeline(pipeline, q)
        our_answered = is_answered(our_ans)

        sib_ans, sib_time = query_sibils(q)
        sib_answered = is_answered(sib_ans)

        our_f1 = squad_f1(our_ans, gold) if our_answered else 0.0
        our_r1 = rouge1(our_ans, gold) if our_answered else 0.0
        our_rl = rouge_l(our_ans, gold) if our_answered else 0.0

        sib_f1 = squad_f1(sib_ans, gold) if sib_answered else 0.0
        sib_r1 = rouge1(sib_ans, gold) if sib_answered else 0.0
        sib_rl = rouge_l(sib_ans, gold) if sib_answered else 0.0

        row = {
            "dataset": name,
            "question": q[:80],
            "golden_answer": gold[:60],
            "our_answer": our_ans[:80],
            "our_answered": int(our_answered),
            "our_f1": round(our_f1, 4),
            "our_rouge1": round(our_r1, 4),
            "our_rougeL": round(our_rl, 4),
            "our_time": round(our_time, 2),
            "sib_answer": sib_ans[:80],
            "sib_answered": int(sib_answered),
            "sib_f1": round(sib_f1, 4),
            "sib_rouge1": round(sib_r1, 4),
            "sib_rougeL": round(sib_rl, 4),
            "sib_time": round(sib_time, 2),
        }
        rows.append(row)
        writer.writerow(row)
        out_file.flush()  # write immediately — survives VM restarts

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(
                f"  [{i+1:3d}/{n}] "
                f"ours: {our_ans[:30]!r:32s} (F1={our_f1:.0%}) | "
                f"sibils: {sib_ans[:30]!r:32s} (F1={sib_f1:.0%})"
            )
    return rows


def print_summary(all_rows: list[dict]):
    import statistics

    datasets = sorted(set(r["dataset"] for r in all_rows))

    header = f"{'Dataset':<14} {'System':<10} {'Answered':>8} {'F1':>7} {'ROUGE-1':>8} {'ROUGE-L':>8} {'Time':>6}"
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(header)
    print("-" * 70)

    for ds in datasets:
        subset = [r for r in all_rows if r["dataset"] == ds]
        for sys, prefix in [("BioMoQA", "our"), ("Sibils v5", "sib")]:
            ans_rate = sum(float(r[f"{prefix}_answered"]) for r in subset) / len(subset)
            avg_f1   = statistics.mean(float(r[f"{prefix}_f1"])     for r in subset)
            avg_r1   = statistics.mean(float(r[f"{prefix}_rouge1"]) for r in subset)
            avg_rl   = statistics.mean(float(r[f"{prefix}_rougeL"]) for r in subset)
            avg_time = statistics.mean(float(r[f"{prefix}_time"])   for r in subset)
            print(
                f"{ds:<14} {sys:<10} {ans_rate:>7.1%} {avg_f1:>7.2%} "
                f"{avg_r1:>8.2%} {avg_rl:>8.2%} {avg_time:>5.1f}s"
            )
        print("-" * 70)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-biomoqa", type=int, default=120)
    parser.add_argument("--limit-bioasq",  type=int, default=50)
    parser.add_argument("--limit-pubmed",  type=int, default=50)
    parser.add_argument("--output", default="results/eval_all.csv")
    parser.add_argument("--json",   default="results/eval_all_summary.json")
    args = parser.parse_args()

    # Load pipeline (extractive only — no GPU needed while API server is running)
    print("\nLoading pipeline (extractive mode, no LLM) …")
    from biomoqa_rag.pipeline import RAGPipeline, RAGConfig
    pipeline = RAGPipeline(RAGConfig(use_vllm=False, use_cpu=False))

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    resume = out.exists()

    # Load already-done rows for resume support
    done_keys = set()
    existing_rows = []
    if resume:
        with open(out) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["dataset"], row["question"]))
                existing_rows.append(row)
        print(f"Resuming: {len(done_keys)} rows already done")

    out_file = open(out, "a" if resume else "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if not resume:
        writer.writeheader()

    all_rows = list(existing_rows)

    # ── Dataset 1: BioMoQA 120 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Dataset 1/3: BioMoQA-120 (parasitology / biodiversity)")
    print("="*60)
    qs = load_biomoqa120(args.limit_biomoqa)
    rows = evaluate_dataset("biomoqa120", qs, pipeline, writer, out_file, done_keys)
    all_rows.extend(rows)

    # ── Dataset 2: BioASQ ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Dataset 2/3: BioASQ factoid (general biomedical QA)")
    print("="*60)
    qs = load_bioasq(args.limit_bioasq)
    rows = evaluate_dataset("bioasq", qs, pipeline, writer, out_file, done_keys)
    all_rows.extend(rows)

    # ── Dataset 3: PubMedQA ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Dataset 3/3: PubMedQA (yes/no/maybe from PubMed abstracts)")
    print("="*60)
    qs = load_pubmedqa(args.limit_pubmed)
    rows = evaluate_dataset("pubmedqa", qs, pipeline, writer, out_file, done_keys)
    all_rows.extend(rows)

    out_file.close()
    print(f"\nDetailed results → {out}")

    print_summary(all_rows)

    # JSON summary
    summary = {}
    for ds in sorted(set(r["dataset"] for r in all_rows)):
        subset = [r for r in all_rows if r["dataset"] == ds]
        summary[ds] = {}
        for sys, prefix in [("biomoqa", "our"), ("sibils_v5", "sib")]:
            n = len(subset)
            summary[ds][sys] = {
                "n": n,
                "answer_rate": round(sum(float(r[f"{prefix}_answered"]) for r in subset) / n, 4),
                "avg_f1":      round(sum(float(r[f"{prefix}_f1"])     for r in subset) / n, 4),
                "avg_rouge1":  round(sum(float(r[f"{prefix}_rouge1"]) for r in subset) / n, 4),
                "avg_rougeL":  round(sum(float(r[f"{prefix}_rougeL"]) for r in subset) / n, 4),
                "avg_time_s":  round(sum(float(r[f"{prefix}_time"])   for r in subset) / n, 2),
            }
    Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON   → {args.json}")
