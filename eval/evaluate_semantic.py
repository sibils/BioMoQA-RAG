"""
Semantic evaluation: extractive vs generative vs Sibils API.

Metrics: SQuAD-F1, ROUGE-1/L, BERTScore-F1, Cosine Similarity (sentence-transformers).

Usage:
    python evaluate_semantic.py                        # both datasets, all 3 systems
    python evaluate_semantic.py --limit 20             # quick smoke test
    python evaluate_semantic.py --resume               # skip already-done rows
    python evaluate_semantic.py --plot-only            # skip eval, just (re)plot
"""

import argparse
import csv
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rouge_score import rouge_scorer

SIBILS_API = "https://biodiversitypmc.sibils.org/api/QA"
OUTPUT_CSV  = Path("results/eval_semantic.csv")
SUMMARY_JSON = Path("results/eval_semantic_summary.json")
PLOT_PATH   = Path("results/eval_semantic_plot.png")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEMS = ["extractive", "generative"]
NO_ANSWER_MARKERS = ["no relevant answer found", "cannot be fully answered", "not found", "no answer"]

# ---------------------------------------------------------------------------
# Lexical metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def squad_f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p)
    rec  = n / len(g)
    return 2 * prec * rec / (prec + rec)


def compute_rouge(pred: str, gold: str) -> dict:
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    s  = sc.score(gold, pred)
    return {"rouge1": s["rouge1"].fmeasure, "rougeL": s["rougeL"].fmeasure}


def is_answered(text: str) -> bool:
    if not text or not text.strip():
        return False
    lo = text.lower()
    return not any(m in lo for m in NO_ANSWER_MARKERS)


# ---------------------------------------------------------------------------
# Semantic metrics (loaded once, lazily)
# ---------------------------------------------------------------------------

_bert_scorer  = None
_sent_model   = None


def _get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        print("  Loading BERTScorer (deberta-xlarge-mnli) …")
        _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device="cuda")
    return _bert_scorer


def _get_sent_model():
    global _sent_model
    if _sent_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence-transformer for cosine sim …")
        _sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    return _sent_model


def bertscore_f1(pred: str, gold: str) -> float:
    if not pred.strip() or not gold.strip():
        return 0.0
    scorer = _get_bert_scorer()
    _, _, F = scorer.score([pred], [gold])
    return float(F[0])


def cosine_sim(pred: str, gold: str) -> float:
    if not pred.strip() or not gold.strip():
        return 0.0
    model = _get_sent_model()
    embs = model.encode([pred, gold], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_biomoqa120(limit=None) -> list[dict]:
    rows = []
    with open("results/biomoqa_120_results.csv") as f:
        for r in csv.DictReader(f):
            rows.append({"dataset": "biomoqa120", "question": r["question"], "golden_answer": r["golden_answer"]})
    return rows[:limit] if limit else rows


def load_bioasq(limit=120) -> list[dict]:
    import re as _re
    from datasets import load_dataset
    print("  Downloading BioASQ …")
    ds = load_dataset("kroshan/BioASQ", split="train")
    rows = []
    for item in ds:
        q    = item.get("question", "").strip()
        text = item.get("text", "")
        m    = _re.search(r"<answer>\s*(.+?)\s*<context>", text, re.DOTALL)
        gold = m.group(1).strip() if m else ""
        if q and gold:
            rows.append({"dataset": "bioasq", "question": q, "golden_answer": gold})
        if len(rows) >= limit:
            break
    print(f"  Loaded {len(rows)} BioASQ questions")
    return rows


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def query_pipeline(pipeline, question: str, mode: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        result = pipeline.run(question=question, mode=mode, return_documents=False)
        text = " ".join(a["answer"] for a in result.get("answers", [])).strip()
    except Exception as e:
        print(f"    pipeline error [{mode}]: {e}")
        text = ""
    return text, time.time() - t0


def query_sibils(question: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        resp = requests.get(SIBILS_API, params={"col": "medline", "q": question, "n": 5}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("answers", [{}])[0].get("answer", "")
    except Exception as e:
        text = ""
    return text, time.time() - t0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "dataset", "question", "golden_answer", "system",
    "answer", "is_answered",
    "f1", "rouge1", "rougeL", "bertscore", "cosine",
    "time_s",
]


def run_eval(args):
    import sys
    sys.path.insert(0, "src")
    from biomoqa_rag.pipeline import RAGPipeline, RAGConfig

    # Load questions
    questions = []
    if "biomoqa120" in args.datasets:
        questions += load_biomoqa120(limit=args.limit)
    if "bioasq" in args.datasets:
        questions += load_bioasq(limit=args.limit or 120)

    # Resume
    done_keys = set()
    if args.resume and OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, on_bad_lines="skip")
        done_keys = set(zip(existing["dataset"], existing["question"].str[:80], existing["system"]))
        print(f"Resuming: {len(done_keys)} rows already done")

    # Load pipeline (needs LLM for generative)
    print("Loading RAG pipeline …")
    pipeline = RAGPipeline(RAGConfig())

    write_header = not OUTPUT_CSV.exists() or not args.resume
    out_file = open(OUTPUT_CSV, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writeheader()

    total = len(questions) * len(SYSTEMS)
    done  = 0

    for item in questions:
        dataset = item["dataset"]
        question = item["question"]
        golden   = item["golden_answer"]

        # Run all 3 systems per question
        answers = {}
        times   = {}
        for system in SYSTEMS:
            key = (dataset, question[:80], system)
            if key in done_keys:
                done += 1
                continue
            if system in ("extractive", "generative"):
                ans, t = query_pipeline(pipeline, question, system)
            else:
                ans, t = query_sibils(question)
            answers[system] = ans
            times[system]   = t

        if not answers:
            continue

        # Compute semantic metrics in batch per question (share encoder calls)
        for system, ans in answers.items():
            answered = is_answered(ans)
            f1    = squad_f1(ans, golden) if answered else 0.0
            rouge = compute_rouge(ans, golden) if answered else {"rouge1": 0.0, "rougeL": 0.0}
            bs    = bertscore_f1(ans, golden) if answered else 0.0
            cs    = cosine_sim(ans, golden)   if answered else 0.0

            writer.writerow({
                "dataset": dataset, "question": question, "golden_answer": golden,
                "system": system, "answer": ans, "is_answered": int(answered),
                "f1": round(f1, 4), "rouge1": round(rouge["rouge1"], 4),
                "rougeL": round(rouge["rougeL"], 4),
                "bertscore": round(bs, 4), "cosine": round(cs, 4),
                "time_s": round(times[system], 2),
            })
            out_file.flush()
            done += 1
            print(f"  [{done}/{total}] {dataset} | {system:11s} | "
                  f"F1={f1:.2%} BS={bs:.2%} cos={cs:.2%} {'Y' if answered else 'N'} {times[system]:.1f}s")

    out_file.close()
    print(f"\nSaved → {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Summary + plot
# ---------------------------------------------------------------------------

def make_summary_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df = pd.read_csv(OUTPUT_CSV, on_bad_lines="skip")
    df["bertscore"] = pd.to_numeric(df["bertscore"], errors="coerce").fillna(0)
    df["cosine"]    = pd.to_numeric(df["cosine"],    errors="coerce").fillna(0)

    metrics = ["f1", "rouge1", "rougeL", "bertscore", "cosine"]
    metric_labels = ["SQuAD F1", "ROUGE-1", "ROUGE-L", "BERTScore", "Cosine Sim"]

    datasets = sorted(df["dataset"].unique())
    systems  = ["extractive", "generative"]
    sys_labels = {"extractive": "Extractive (BioBERT)", "generative": "Generative (Qwen3-8B)"}
    colors = {"extractive": "#4C72B0", "generative": "#DD8452"}

    summary_rows = []
    for ds in datasets:
        for sys in systems:
            sub = df[(df["dataset"] == ds) & (df["system"] == sys)]
            if sub.empty:
                continue
            row = {"dataset": ds, "system": sys, "n": len(sub),
                   "answer_rate": sub["is_answered"].mean()}
            for m in metrics:
                row[m] = sub[m].mean()
            row["avg_time"] = sub["time_s"].mean()
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_json(SUMMARY_JSON, orient="records", indent=2)
    print(f"Summary → {SUMMARY_JSON}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_datasets = len(datasets)
    n_metrics  = len(metrics)
    fig, axes  = plt.subplots(n_datasets, n_metrics,
                              figsize=(5 * n_metrics, 4.5 * n_datasets),
                              sharey=False)
    if n_datasets == 1:
        axes = [axes]

    bar_width = 0.25
    x = np.arange(len(systems))

    for di, ds in enumerate(datasets):
        for mi, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[di][mi]
            sub = summary[summary["dataset"] == ds]
            for xi, sys in enumerate(systems):
                row = sub[sub["system"] == sys]
                val = float(row[metric].iloc[0]) if not row.empty else 0.0
                ax.bar(xi, val, color=colors[sys], width=0.6, alpha=0.85,
                       label=sys_labels[sys] if di == 0 and mi == 0 else "")
                ax.text(xi, val + 0.005, f"{val:.2%}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")

            ax.set_xticks(range(len(systems)))
            ax.set_xticklabels([sys_labels[s] for s in systems], rotation=15, ha="right", fontsize=8)
            ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.25))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.set_title(f"{label}", fontsize=11, fontweight="bold")
            if mi == 0:
                ax.set_ylabel(ds.upper(), fontsize=10, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legend
    patches = [mpatches.Patch(color=colors[s], label=sys_labels[s]) for s in systems]
    fig.legend(handles=patches, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle("BioMoQA-RAG: Semantic Evaluation\nExtractive (BioBERT) vs Generative (Qwen3-8B)",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Plot    → {PLOT_PATH}")

    # Print table
    print("\n" + "=" * 80)
    print("SEMANTIC EVALUATION SUMMARY")
    print("=" * 80)
    display = summary.copy()
    for m in metrics:
        display[m] = display[m].map(lambda v: f"{v:.2%}")
    display["answer_rate"] = display["answer_rate"].map(lambda v: f"{v:.2%}")
    display["avg_time"]    = display["avg_time"].map(lambda v: f"{v:.2f}s")
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["biomoqa120", "bioasq"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip evaluation, just regenerate plot from existing CSV")
    args = parser.parse_args()

    if not args.plot_only:
        run_eval(args)

    make_summary_and_plot()
