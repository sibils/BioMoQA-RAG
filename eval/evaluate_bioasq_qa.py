"""
BioASQ QA-only evaluation — retrieval bypassed.

Uses the gold passage from each BioASQ item directly as context, bypassing
retrieval entirely. Measures pure QA reader quality: extractive (BioBERT)
and generative (Qwen3-8B).

This eliminates retrieval bias: BioASQ questions come from PubMed abstracts,
and a live retrieval eval would unfairly reward the system when SIBILS
happens to return the exact source document.

Dataset: kroshan/BioASQ on HuggingFace (split="train")
  Each item: question + text field structured as:
  <answer>GOLD_ANSWER <context>GOLD_CONTEXT</context>

Metrics:
  - Exact Match      — normalized string equality (primary for factoid)
  - Answer Contains  — gold answer is a substring of the generated answer
                       (more appropriate than EM for verbose generative output)
  - SQuAD F1         — token overlap, partial credit
  - ROUGE-1 / ROUGE-L
  - BERTScore        — semantic similarity

Usage:
    python evaluate_bioasq_qa.py --limit 5 --extractive-only   # smoke test
    python evaluate_bioasq_qa.py --limit 200 --extractive-only # full extractive (~5 min)
    python evaluate_bioasq_qa.py --limit 200                   # extractive + generative (~30 min)
    python evaluate_bioasq_qa.py --plot-only                   # regenerate plot from CSV
"""

import argparse
import csv
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

OUTPUT_CSV   = Path("results/eval_bioasq_qa.csv")
SUMMARY_JSON = Path("results/eval_bioasq_qa_summary.json")
PLOT_PATH    = Path("results/eval_bioasq_qa_plot.png")
Path("results").mkdir(exist_ok=True)

NO_ANSWER_MARKERS = [
    "no relevant answer found", "cannot be fully answered",
    "not found", "no answer", "insufficient",
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def answer_contains(pred: str, gold: str) -> float:
    """Gold answer (normalized) appears as a substring of the prediction."""
    return float(_normalize(gold) in _normalize(pred))


def squad_f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
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


_bert_scorer = None
_sent_model  = None


def _get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        print("  Loading BERTScorer …")
        _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device="cpu")
    return _bert_scorer


def bertscore_f1(pred: str, gold: str) -> float:
    if not pred.strip() or not gold.strip():
        return 0.0
    _, _, F = _get_bert_scorer().score([pred], [gold])
    return float(F[0])


def faithfulness_score(pred: str, context: str) -> float:
    """Sentence-level grounding score (TREC AIS proxy).

    For each sentence of the generated answer, compute ROUGE-1 recall against
    the gold context (fraction of the sentence's tokens that appear in the
    context). Return the mean across sentences.

    High score (>0.6): answer is predominantly grounded in the context.
    Low score (<0.3): answer introduces content not present in the context
                      (potential hallucination or off-topic hedge).
    """
    if not pred.strip() or not context.strip():
        return 0.0
    pred_clean = re.sub(r"^Based on the provided documents?,?\s*", "", pred, flags=re.IGNORECASE)
    sentences = [s.strip() for s in re.split(r"[.!?]", pred_clean)
                 if s.strip() and len(s.strip().split()) >= 3]
    if not sentences:
        return 0.0
    ctx_tokens = Counter(_normalize(context).split())
    recalls = []
    for sent in sentences:
        toks = _normalize(sent).split()
        if not toks:
            continue
        common = Counter(toks) & ctx_tokens
        recalls.append(sum(common.values()) / len(toks))
    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bioasq(limit: int | None = None) -> list[dict]:
    print("  Downloading BioASQ from HuggingFace …")
    from datasets import load_dataset
    ds = load_dataset("kroshan/BioASQ", split="train")
    seen = set()
    rows = []
    for item in ds:
        q    = item.get("question", "").strip()
        text = item.get("text", "")
        m_ans = re.search(r"<answer>\s*(.+?)\s*<context>", text, re.DOTALL)
        m_ctx = re.search(r"<context>\s*(.+?)(?:</context>|$)", text, re.DOTALL)
        gold    = m_ans.group(1).strip() if m_ans else ""
        context = m_ctx.group(1).strip() if m_ctx else ""
        if q and gold and context and q not in seen:
            seen.add(q)
            rows.append({"question": q, "golden_answer": gold, "gold_context": context})
        if limit and len(rows) >= limit:
            break
    print(f"  Loaded {len(rows)} questions with gold context")
    return rows


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline(extractive_only: bool = False):
    global _pipeline
    if _pipeline is None:
        import sys
        sys.path.insert(0, "src")
        from biomoqa_rag.pipeline import RAGPipeline, RAGConfig
        cfg = RAGConfig(use_vllm=not extractive_only, use_cpu=False)
        print("Loading pipeline …")
        _pipeline = RAGPipeline(cfg)
        print("Pipeline ready.")
    return _pipeline


def run_system(pipeline, question: str, context: str, mode: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        result = pipeline.run_with_contexts(question=question, contexts=[context], mode=mode)
        answers = result.get("answers", []) or []
        if not answers:
            return "", time.time() - t0
        text = answers[0].get("answer", "").strip()
        return text, time.time() - t0
    except Exception as e:
        print(f"    [{mode} error] {e}")
        return "", time.time() - t0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "question_id", "question", "golden_answer", "gold_context",
    "system", "answer", "is_answered",
    "exact_match", "answer_contains",
    "f1", "rouge1", "rougeL", "bertscore",
    "faithfulness",
    "time_s",
]


def run_eval(args):
    rows    = load_bioasq(limit=args.limit)
    systems = ["extractive"] if args.extractive_only else ["extractive", "generative"]

    done_keys = set()
    if args.resume and args.output.exists():
        existing  = pd.read_csv(args.output, on_bad_lines="skip")
        done_keys = set(zip(existing["question"].str[:80], existing["system"]))
        print(f"Resuming: {len(done_keys)} rows already done")

    pipeline = get_pipeline(extractive_only=args.extractive_only)

    write_header = not args.output.exists() or not args.resume
    out_file = open(args.output, "a" if args.resume else "w", newline="")
    writer   = csv.DictWriter(out_file, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writeheader()

    total = len(rows) * len(systems)
    done  = 0

    for i, row in enumerate(rows):
        qid      = i + 1
        question = row["question"]
        golden   = row["golden_answer"]
        context  = row["gold_context"]

        print(f"\n[{qid}/{len(rows)}] {question[:80]}")
        print(f"  gold: {golden}")

        for system in systems:
            key = (question[:80], system)
            if key in done_keys:
                done += 1
                continue

            ans, elapsed = run_system(pipeline, question, context, system)
            answered = is_answered(ans)

            f1    = squad_f1(ans, golden)       if answered else 0.0
            em    = exact_match(ans, golden)     if answered else 0.0
            ac    = answer_contains(ans, golden) if answered else 0.0
            rouge = compute_rouge(ans, golden)   if answered else {"rouge1": 0.0, "rougeL": 0.0}
            bs    = bertscore_f1(ans, golden)    if answered else 0.0
            faith = faithfulness_score(ans, context) if answered and system == "generative" else float("nan")

            print(
                f"  [{system:11s}] {elapsed:.1f}s | "
                f"EM={em:.0%} Contains={ac:.0%} F1={f1:.2%} BS={bs:.2%} | "
                f"{ans[:70] if answered else '(no answer)'}"
            )

            writer.writerow({
                "question_id":     qid,
                "question":        question,
                "golden_answer":   golden,
                "gold_context":    context,
                "system":          system,
                "answer":          ans,
                "is_answered":     int(answered),
                "exact_match":     round(em, 4),
                "answer_contains": round(ac, 4),
                "f1":              round(f1, 4),
                "rouge1":          round(rouge["rouge1"], 4),
                "rougeL":          round(rouge["rougeL"], 4),
                "bertscore":       round(bs, 4),
                "faithfulness":    round(faith, 4) if not (isinstance(faith, float) and faith != faith) else "",
                "time_s":          round(elapsed, 2),
            })
            out_file.flush()
            done += 1

    out_file.close()
    print(f"\nSaved → {args.output}")


# ---------------------------------------------------------------------------
# Summary + plot
# ---------------------------------------------------------------------------

def make_summary_and_plot(output_csv: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df = pd.read_csv(output_csv, on_bad_lines="skip")
    for col in ["bertscore", "f1", "rouge1", "rougeL", "exact_match", "answer_contains"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    metrics       = ["exact_match", "answer_contains", "f1", "rouge1", "rougeL", "bertscore"]
    metric_labels = ["Exact Match", "Answer Contains", "SQuAD F1", "ROUGE-1", "ROUGE-L", "BERTScore"]
    df["faithfulness"] = pd.to_numeric(df.get("faithfulness", float("nan")), errors="coerce")
    systems       = sorted(df["system"].unique())
    sys_labels    = {"extractive": "Extractive (BioBERT)", "generative": "Generative (Qwen3-8B)"}
    colors        = {"extractive": "#4C72B0", "generative": "#DD8452"}

    summary_rows = []
    for sys in systems:
        sub = df[df["system"] == sys]
        if sub.empty:
            continue
        row = {"system": sys, "n": len(sub), "answer_rate": sub["is_answered"].mean()}
        for m in metrics:
            row[m] = sub[m].mean()
        if sys == "generative" and "faithfulness" in sub.columns:
            row["faithfulness"] = sub["faithfulness"].mean()
        row["avg_time"] = sub["time_s"].mean()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_json(SUMMARY_JSON, orient="records", indent=2)
    print(f"Summary → {SUMMARY_JSON}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 5), sharey=False)
    for mi, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[mi]
        for xi, sys in enumerate(systems):
            row = summary[summary["system"] == sys]
            val = float(row[metric].iloc[0]) if not row.empty else 0.0
            ax.bar(xi, val, color=colors.get(sys, "#888"), width=0.6, alpha=0.85)
            ax.text(xi, val + 0.008, f"{val:.1%}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(
            [sys_labels.get(s, s) for s in systems], rotation=15, ha="right", fontsize=8
        )
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.3))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    patches = [
        mpatches.Patch(color=colors.get(s, "#888"), label=sys_labels.get(s, s))
        for s in systems
    ]
    fig.legend(handles=patches, loc="upper center", ncol=len(systems), fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        "BioASQ QA-only benchmark (gold context — retrieval bypassed)\n"
        "Pure QA reader quality: BioBERT vs Qwen3-8B",
        fontsize=11, fontweight="bold", y=1.1,
    )
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Plot    → {PLOT_PATH}")

    print("\n" + "=" * 70)
    print("BioASQ QA-ONLY EVALUATION SUMMARY (retrieval bypassed)")
    print("=" * 70)
    display = summary.copy()
    for m in metrics:
        display[m] = display[m].map(lambda v: f"{v:.1%}")
    display["answer_rate"] = display["answer_rate"].map(lambda v: f"{v:.1%}")
    display["avg_time"]    = display["avg_time"].map(lambda v: f"{v:.2f}s")
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BioASQ QA-only eval — gold context supplied, retrieval bypassed"
    )
    parser.add_argument("--limit", type=int, default=200,
                        help="Number of BioASQ questions (default: 200)")
    parser.add_argument("--extractive-only", action="store_true",
                        help="Skip generative mode (no GPU/vLLM required)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the output CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip eval, regenerate plot/summary from existing CSV")
    args = parser.parse_args()

    if not args.plot_only:
        run_eval(args)

    make_summary_and_plot(args.output)
