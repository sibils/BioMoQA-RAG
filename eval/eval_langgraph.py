"""
Benchmark: RAGPipeline.run() (original) vs LangGraphRAGPipeline.run() (LangGraph).

Loads ONE pipeline instance (LangGraphRAGPipeline, which inherits all components).
Compares orchestration overhead by calling both run() implementations on the same model:
  - "original"  → RAGPipeline.run(self, ...)   via super()
  - "langgraph" → self.run(...)                via LangGraph graph.invoke()

This isolates pure orchestration cost from LLM inference time.

Usage (from repo root):
    python eval/eval_langgraph.py
    python eval/eval_langgraph.py --queries 10 --mode generative
    python eval/eval_langgraph.py --mode extractive
    python eval/eval_langgraph.py --queries 3 --mode generative  # quick smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

QUERIES = [
    "What is the mechanism of action of hydroxychloroquine in malaria?",
    "What causes Lyme disease and how is it transmitted?",
    "What are the main symptoms of COVID-19?",
    "How does CRISPR-Cas9 gene editing work?",
    "What is the role of p53 in cancer suppression?",
    "What vaccines are effective against tick-borne encephalitis?",
    "What is the treatment for Type 2 diabetes?",
    "How does antibiotic resistance develop in bacteria?",
    "What are the risk factors for cardiovascular disease?",
    "What is the pathogenesis of Alzheimer's disease?",
]


def run_original(pipeline, queries, mode, collection):
    """Call RAGPipeline.run() directly, bypassing the LangGraph override."""
    from biomoqa_rag.pipeline import RAGPipeline
    results = []
    for q in queries:
        t0 = time.time()
        r = RAGPipeline.run(pipeline, q, collection=collection, mode=mode,
                            retrieval_n=10, final_n=5)
        elapsed = time.time() - t0
        answer_text = r["answers"][0]["answer"] if r["answers"] else "(no answer)"
        results.append({
            "question": q,
            "answer": answer_text,
            "latency": round(elapsed, 3),
            "ndocs": r["ndocs_returned_by_SIBiLS"],
        })
    return results


def run_langgraph(pipeline, queries, mode, collection):
    """Call the overridden run() — goes through LangGraph graph.invoke()."""
    results = []
    for q in queries:
        t0 = time.time()
        r = pipeline.run(q, collection=collection, mode=mode,
                         retrieval_n=10, final_n=5)
        elapsed = time.time() - t0
        answer_text = r["answers"][0]["answer"] if r["answers"] else "(no answer)"
        results.append({
            "question": q,
            "answer": answer_text,
            "latency": round(elapsed, 3),
            "ndocs": r["ndocs_returned_by_SIBiLS"],
        })
    return results


def print_answers(label, results):
    print(f"\n{'─'*72}")
    print(f"  {label} answers")
    print(f"{'─'*72}")
    for r in results:
        print(f"\n  Q: {r['question']}")
        print(f"  A: {r['answer'][:250]}")
        print(f"     [{r['latency']}s | {r['ndocs']} docs retrieved]")


def compare(orig_results, lg_results):
    print("\n" + "=" * 74)
    print(f"{'Query':<46} {'Orig (s)':>8} {'LG (s)':>8} {'Δ (ms)':>8} {'Match':>4}")
    print("─" * 74)
    latency_orig, latency_lg, mismatches = [], [], []
    for o, l in zip(orig_results, lg_results):
        delta_ms = round((l["latency"] - o["latency"]) * 1000)
        same = o["answer"].strip() == l["answer"].strip()
        if not same:
            mismatches.append((o["question"], o["answer"], l["answer"]))
        q_short = o["question"][:45]
        marker = "✓" if same else "✗"
        print(f"{q_short:<46} {o['latency']:>8.2f} {l['latency']:>8.2f} {delta_ms:>+8} {marker:>4}")
        latency_orig.append(o["latency"])
        latency_lg.append(l["latency"])

    n = len(latency_orig)
    avg_orig = sum(latency_orig) / n
    avg_lg   = sum(latency_lg) / n
    avg_delta = round((avg_lg - avg_orig) * 1000)
    print("─" * 74)
    print(f"{'AVERAGE':<46} {avg_orig:>8.2f} {avg_lg:>8.2f} {avg_delta:>+8}")
    print()

    print(f"Queries : {n}")
    print(f"Matches : {n - len(mismatches)}/{n}  {'(identical output)' if not mismatches else ''}")
    print(f"Overhead: {avg_delta:+d} ms average per query  "
          f"({round(avg_delta / avg_orig * 100, 1):+.1f}% of original latency)")

    if mismatches:
        print("\nMismatched answers:")
        for q, a_orig, a_lg in mismatches:
            print(f"  Q   : {q}")
            print(f"  orig: {a_orig[:120]}")
            print(f"  lg  : {a_lg[:120]}")

    return {
        "n_queries": n,
        "avg_orig_s": round(avg_orig, 3),
        "avg_lg_s":   round(avg_lg, 3),
        "avg_overhead_ms": avg_delta,
        "overhead_pct": round(avg_delta / avg_orig * 100, 1),
        "answer_matches": n - len(mismatches),
        "answer_mismatches": len(mismatches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries",    type=int, default=5,
                        help="Number of queries to run (max 10, default 5)")
    parser.add_argument("--mode",       default="generative",
                        choices=["generative", "extractive"])
    parser.add_argument("--collection", default="medline")
    args = parser.parse_args()

    queries = QUERIES[:min(args.queries, len(QUERIES))]
    print(f"\nBenchmark: original RAGPipeline.run() vs LangGraph run()")
    print(f"Mode: {args.mode} | Collection: {args.collection} | Queries: {len(queries)}\n")

    from biomoqa_rag.pipeline import RAGConfig
    from biomoqa_rag.pipeline_langgraph import LangGraphRAGPipeline

    config = RAGConfig()  # Qwen3-8B + vLLM + FP8
    print("Loading LangGraphRAGPipeline (shared instance for both runs)...")
    pipeline = LangGraphRAGPipeline(config)

    # Warm-up: one query so vLLM JIT-compiles before the timed runs
    print("\nWarming up vLLM...")
    _ = pipeline.run(QUERIES[0], collection=args.collection, mode=args.mode,
                     retrieval_n=5, final_n=3)
    print("Warm-up done.\n")

    print("── Run 1: original RAGPipeline.run() ──")
    orig_results = run_original(pipeline, queries, args.mode, args.collection)

    print("── Run 2: LangGraphRAGPipeline.run() ──")
    lg_results = run_langgraph(pipeline, queries, args.mode, args.collection)

    print_answers("original", orig_results)
    print_answers("langgraph", lg_results)

    summary = compare(orig_results, lg_results)
    print(f"\nJSON summary:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
