"""
Test different FAISS index options and alpha values for RAG retrieval.

Runs the same 50 BioASQ questions with:
  - alpha=0.3 (more BM25 weight)
  - alpha=0.5 (current default, equal weight)
  - alpha=0.7 (more FAISS weight)
  - PubMedBERT index vs MiniLM index (at alpha=0.5)

Usage:
    python evaluate_alpha.py --limit 50
"""

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path


def _normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def squad_f1(prediction, ground_truth):
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


def context_recall(gold, documents):
    gold_norm = _normalize(gold)
    for doc in documents:
        text = ((doc.title or "") + " " + (doc.abstract or "")).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        if gold_norm in text:
            return True
    return False


def load_bioasq(limit=None):
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
    return rows


def run_config(questions, sibils_retriever, dense_retriever, alpha, extractor, cfg):
    """Run one alpha configuration over all questions."""
    from src.biomoqa_rag.retrieval.parallel_hybrid import ParallelHybridRetriever

    hybrid = ParallelHybridRetriever(sibils_retriever, dense_retriever, alpha=alpha, k=60)

    totals = {"ctx_recall": 0, "answered": 0, "f1": 0.0, "ndocs": 0, "time": 0.0}

    for row in questions:
        question = row["question"]
        gold = row["golden_answer"]
        t0 = time.time()
        try:
            docs = hybrid.retrieve(question, n=cfg["retrieval_n"], top_k=cfg["final_n"])
            docs = docs[:cfg["final_n"]]
        except Exception:
            docs = []
        answer = ""
        if docs:
            candidates = extractor.extract(question, docs, 2000)
            if candidates:
                answer = candidates[0]["text"].strip()
        elapsed = time.time() - t0
        cr = context_recall(gold, docs)
        f1 = squad_f1(answer, gold) if answer else 0.0
        totals["ctx_recall"] += int(cr)
        totals["answered"] += int(bool(answer))
        totals["f1"] += f1
        totals["ndocs"] += len(docs)
        totals["time"] += elapsed

    n = len(questions)
    return {k: round(v / n, 3) if k not in ("ctx_recall", "answered") else round(v / n * 100, 1)
            for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="results/alpha_eval.json")
    args = parser.parse_args()

    print("Loading BioASQ…")
    rows = load_bioasq(args.limit)
    n = len(rows)
    print(f"  {n} questions")

    from src.biomoqa_rag.retrieval.sibils_retriever import SIBILSRetriever
    from src.biomoqa_rag.retrieval.dense_retriever import DenseRetriever
    from src.biomoqa_rag.extraction.extractive_qa import BioExtractiveQA

    print("Loading components…")
    sibils = SIBILSRetriever(
        collection=["medline", "plazi", "pmc"],
        cache_dir="data/sibils_cache",
        cache_ttl=604800,
    )
    extractor = BioExtractiveQA(device=-1)
    cfg = {"retrieval_n": 10, "final_n": 5}

    # Each tuple: (label, alpha, faiss_index, docs_pkl, query_encoder_model)
    # IMPORTANT: query encoder must match the model used to build the index.
    configs = [
        # Current production index: S-PubMedBert-MS-MARCO (biomedical retrieval-tuned)
        ("PubMedBert-MSMARCO alpha=0.3", 0.3, "data/faiss_index.bin", "data/documents.pkl",
         "pritamdeka/S-PubMedBert-MS-MARCO"),
        ("PubMedBert-MSMARCO alpha=0.5", 0.5, "data/faiss_index.bin", "data/documents.pkl",
         "pritamdeka/S-PubMedBert-MS-MARCO"),
        ("PubMedBert-MSMARCO alpha=0.7", 0.7, "data/faiss_index.bin", "data/documents.pkl",
         "pritamdeka/S-PubMedBert-MS-MARCO"),
    ]

    # Add old MiniLM backup index if it exists (built before migration)
    miniLM_backup = Path("data/faiss_index.bin.bak")
    if miniLM_backup.exists():
        configs.insert(0, (
            "MiniLM alpha=0.3 (old)", 0.3,
            str(miniLM_backup),
            "data/documents.pkl.bak",
            "sentence-transformers/all-MiniLM-L6-v2",
        ))

    results = {}
    for label, alpha, idx_path, docs_path, model_name in configs:
        print(f"\n── {label} ──")
        dense = DenseRetriever(model_name=model_name)
        dense.load(idx_path, docs_path)
        r = run_config(rows, sibils, dense, alpha, extractor, cfg)
        results[label] = r
        print(f"  ctx_recall={r['ctx_recall']}%  F1={r['f1']:.3f}  answered={r['answered']}%  "
              f"ndocs={r['ndocs']:.1f}  time={r['time']:.2f}s")

    print("\n" + "=" * 70)
    print(f"{'Config':<30} {'Recall':>8} {'F1':>8} {'Answered':>10} {'Time(s)':>8}")
    print("-" * 70)
    for label, r in results.items():
        print(f"{label:<30} {r['ctx_recall']:>7}% {r['f1']:>8.3f} {r['answered']:>9}% {r['time']:>8.2f}")

    Path(args.output).parent.mkdir(exist_ok=True)
    Path(args.output).write_text(json.dumps({"n": n, "configs": results}, indent=2))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
