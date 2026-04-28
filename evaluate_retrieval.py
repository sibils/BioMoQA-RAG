"""
Evaluate retrieval quality: SIBILS (BM25-only) vs RAG (FAISS+BM25+reranker).

Metrics per retrieval strategy:
  context_recall  — does any retrieved doc contain the gold answer string?
  f1 / rouge1     — extractive end-to-end (BioBERT on retrieved docs)
  ndocs           — avg number of docs actually returned
  avg_time        — retrieval + extraction latency

Also reports doc source distribution (medline/plazi/pmc) and overlap between
the two retrieval strategies (how many same docids appear in both?).

Usage:
    python evaluate_retrieval.py --limit 50
    python evaluate_retrieval.py --limit 100 --output results/retrieval_eval.csv
    python evaluate_retrieval.py --limit 50 --collections medline
"""

import argparse
import csv
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
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
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)


def context_recall(gold: str, documents) -> bool:
    """True if any retrieved doc text contains the gold answer (case-insensitive substring)."""
    gold_norm = _normalize(gold)
    for doc in documents:
        text = ((doc.title or "") + " " + (doc.abstract or "")).lower()
        text = " ".join(text.split())
        # remove punctuation from doc text too
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        if gold_norm in text:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# Pipeline loading (lazy, shared)
# ─────────────────────────────────────────────────────────────

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.biomoqa_rag.pipeline import RAGPipeline, RAGConfig
        print("  Loading pipeline (no LLM — extractive only)…")
        _pipeline = RAGPipeline(RAGConfig(use_vllm=False, use_cpu=False))
        print("  Pipeline ready.")
    return _pipeline


# ─────────────────────────────────────────────────────────────
# Per-retrieval runner
# ─────────────────────────────────────────────────────────────

def run_one(question: str, retrieval: str, collection=None) -> dict:
    """
    Retrieve docs then run extractive BioBERT.
    Returns {answer, docs, ndocs, time}.
    """
    p = get_pipeline()
    cfg = p.config
    t0 = time.time()

    docs, _ = p._retrieve_and_prepare(
        question,
        retrieval_n=cfg.retrieval_n,
        final_n=cfg.final_n,
        collection=collection,
        retrieval=retrieval,
    )

    answer = ""
    if docs:
        candidates = p.extractor.extract(question, docs, cfg.max_abstract_length)
        if candidates:
            answer = candidates[0]["text"].strip()

    elapsed = time.time() - t0
    return {"answer": answer, "docs": docs, "ndocs": len(docs), "time": elapsed}


# ─────────────────────────────────────────────────────────────
# BioASQ loader
# ─────────────────────────────────────────────────────────────

def load_bioasq(limit=None):
    print("  Downloading BioASQ from HuggingFace…")
    from datasets import load_dataset
    ds = load_dataset("kroshan/BioASQ", split="train")
    seen, rows = set(), []
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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of BioASQ questions (default: 50)")
    parser.add_argument("--output", default="results/retrieval_eval.csv")
    parser.add_argument("--collections", default=None,
                        help="Comma-separated collections, e.g. 'medline' or None for all")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    collection = args.collections  # None = default (medline+plazi+pmc)

    rows = load_bioasq(limit=args.limit)
    n = len(rows)
    print(f"\nEvaluating {n} questions — SIBILS vs RAG retrieval (extractive)")
    print(f"Collection filter: {collection or 'all (medline+plazi+pmc)'}")
    print("=" * 70)

    strategies = ["sibils", "rag"]
    totals = {s: {"ctx_recall": 0, "answered": 0, "f1": 0.0, "rouge1": 0.0,
                  "ndocs": 0, "time": 0.0,
                  "sources": defaultdict(int), "overlap_count": 0}
              for s in strategies}

    fieldnames = [
        "qid", "question", "golden_answer",
        "sibils_answer", "rag_answer",
        "sibils_ctx_recall", "rag_ctx_recall",
        "sibils_f1", "rag_f1",
        "sibils_ndocs", "rag_ndocs",
        "sibils_time", "rag_time",
        "docid_overlap",          # number of docids shared between both strategies
        "sibils_sources",         # e.g. "medline:3,plazi:1,pmc:1"
        "rag_sources",
    ]

    all_results = []

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, row in enumerate(rows):
            qid = i + 1
            question = row["question"]
            gold = row["golden_answer"]

            print(f"\n[{qid}/{n}] {question[:75]}")
            print(f"  gold: {gold[:60]}")

            results = {}
            for strategy in strategies:
                try:
                    results[strategy] = run_one(question, strategy, collection)
                except Exception as e:
                    print(f"    [{strategy} error] {e}")
                    results[strategy] = {"answer": "", "docs": [], "ndocs": 0, "time": 0.0}

            # Docid overlap
            def docids(docs):
                out = set()
                for d in docs:
                    pmid = getattr(d, "pmid", None)
                    if pmid:
                        out.add(str(pmid))
                    else:
                        out.add(getattr(d, "doc_id", None) or "")
                return out

            sibils_ids = docids(results["sibils"]["docs"])
            rag_ids = docids(results["rag"]["docs"])
            overlap = len(sibils_ids & rag_ids)

            record = {"qid": qid, "question": question, "golden_answer": gold}
            for strategy in strategies:
                r = results[strategy]
                ans = r["answer"]
                cr = context_recall(gold, r["docs"])
                f1 = squad_f1(ans, gold) if ans else 0.0

                src_counter = Counter(getattr(d, "source", "?") for d in r["docs"])
                src_str = ",".join(f"{k}:{v}" for k, v in sorted(src_counter.items()))

                totals[strategy]["ctx_recall"] += int(cr)
                totals[strategy]["answered"] += int(bool(ans))
                totals[strategy]["f1"] += f1
                totals[strategy]["ndocs"] += r["ndocs"]
                totals[strategy]["time"] += r["time"]
                for k, v in src_counter.items():
                    totals[strategy]["sources"][k] += v

                record[f"{strategy}_answer"] = ans
                record[f"{strategy}_ctx_recall"] = cr
                record[f"{strategy}_f1"] = round(f1, 4)
                record[f"{strategy}_ndocs"] = r["ndocs"]
                record[f"{strategy}_time"] = round(r["time"], 2)
                record[f"{strategy}_sources"] = src_str

                print(f"  {strategy:6s} ({r['time']:.1f}s, {r['ndocs']} docs, recall={cr}): {ans[:60] or '(none)'}")

            record["docid_overlap"] = overlap
            all_results.append(record)
            writer.writerow(record)
            f.flush()

            print(f"  F1: sibils={record['sibils_f1']:.3f}  rag={record['rag_f1']:.3f}  overlap={overlap}/{min(len(sibils_ids), len(rag_ids))}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Metric':<28} {'SIBILS (BM25)':>18} {'RAG (FAISS+BM25)':>18}")
    print("-" * 66)

    def pct(k, s): return f"{totals[s][k]/n*100:.1f}%"
    def avg(k, s): return f"{totals[s][k]/n:.3f}"
    def avgf(k, s): return f"{totals[s][k]/n:.1f}"

    print(f"{'Context recall':<28} {pct('ctx_recall','sibils'):>18} {pct('ctx_recall','rag'):>18}")
    print(f"{'Answer rate':<28} {pct('answered','sibils'):>18} {pct('answered','rag'):>18}")
    print(f"{'F1 (avg)':<28} {avg('f1','sibils'):>18} {avg('f1','rag'):>18}")
    print(f"{'Avg docs returned':<28} {avgf('ndocs','sibils'):>18} {avgf('ndocs','rag'):>18}")
    print(f"{'Avg time (s)':<28} {avg('time','sibils'):>18} {avg('time','rag'):>18}")

    print("\nSource distribution (total docs):")
    for strategy in strategies:
        src = totals[strategy]["sources"]
        total_src = sum(src.values())
        dist = "  ".join(f"{k}: {v} ({v/total_src*100:.0f}%)" for k, v in sorted(src.items()))
        print(f"  {strategy}: {dist}")

    summary = {
        "n_questions": n,
        "collection_filter": collection,
        "sibils": {
            "context_recall_pct": round(totals["sibils"]["ctx_recall"] / n * 100, 1),
            "answer_rate_pct": round(totals["sibils"]["answered"] / n * 100, 1),
            "f1": round(totals["sibils"]["f1"] / n, 4),
            "avg_ndocs": round(totals["sibils"]["ndocs"] / n, 1),
            "avg_time": round(totals["sibils"]["time"] / n, 2),
            "source_distribution": dict(totals["sibils"]["sources"]),
        },
        "rag": {
            "context_recall_pct": round(totals["rag"]["ctx_recall"] / n * 100, 1),
            "answer_rate_pct": round(totals["rag"]["answered"] / n * 100, 1),
            "f1": round(totals["rag"]["f1"] / n, 4),
            "avg_ndocs": round(totals["rag"]["ndocs"] / n, 1),
            "avg_time": round(totals["rag"]["time"] / n, 2),
            "source_distribution": dict(totals["rag"]["sources"]),
        },
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
