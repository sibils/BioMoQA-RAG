#!/usr/bin/env python3
"""
Test V2 Improvements

Compare V1 vs V2 on sample questions to verify improvements.
"""

import time
from src.pipeline_vllm import FastRAGPipeline, RAGConfig
from src.pipeline_vllm_v2 import EnhancedRAGPipeline, RAGConfigV2


# Test questions
TEST_QUESTIONS = [
    "What is AG1-IA?",
    "What causes corn sheath blight?",
    "What is the host of Plasmodium falciparum?",
    "What are the common names of Pthirus pubis?",
    "What is the host of V. parahaemolyticus?",
]


def test_v1():
    """Test V1 pipeline."""
    print("="*80)
    print("Testing V1 Pipeline")
    print("="*80)

    config = RAGConfig(
        retrieval_n=50,
        rerank_n=20,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_vllm=True,
        gpu_memory_utilization=0.4,  # Lower to leave room for V2
    )

    pipeline = FastRAGPipeline(config)

    results = []
    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        start = time.time()

        result = pipeline.run(question, return_documents=True)

        elapsed = time.time() - start

        answer_text = " ".join([s["text"] for s in result["answer"]])
        print(f"A: {answer_text[:200]}...")
        print(f"Time: {elapsed:.2f}s")
        print(f"Docs retrieved: {result['num_retrieved']}")

        results.append({
            "question": question,
            "answer": answer_text,
            "time": elapsed,
            "num_docs": result['num_retrieved'],
            "version": "v1"
        })

    return results


def test_v2():
    """Test V2 pipeline."""
    print("\n\n")
    print("="*80)
    print("Testing V2 Pipeline (Enhanced)")
    print("="*80)

    config = RAGConfigV2(
        retrieval_n=100,
        use_query_expansion=True,
        n_query_variants=1,
        use_reranking=True,
        rerank_n=30,
        use_relevance_filter=True,
        relevance_filter_type="fast",
        final_n=20,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_vllm=True,
        gpu_memory_utilization=0.4,
    )

    pipeline = EnhancedRAGPipeline(config)

    results = []
    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        start = time.time()

        result = pipeline.run(question, return_documents=True, debug=True)

        elapsed = time.time() - start

        answer_text = " ".join([s["text"] for s in result["answer"]])
        print(f"A: {answer_text[:200]}...")
        print(f"Time: {elapsed:.2f}s")
        print(f"Docs retrieved: {result['num_retrieved']}")

        if "debug_info" in result:
            debug = result["debug_info"]
            print(f"Debug:")
            print(f"  Queries expanded: {len(debug.get('expanded_queries', []))}")
            print(f"  Initial retrieval: {debug.get('initial_retrieval_count', 0)}")
            print(f"  After reranking: {debug.get('reranked_count', 0)}")
            print(f"  After filtering: {debug.get('filtered_count', 0)}")

        results.append({
            "question": question,
            "answer": answer_text,
            "time": elapsed,
            "num_docs": result['num_retrieved'],
            "version": "v2",
            "debug": result.get("debug_info", {})
        })

    return results


def compare_results(v1_results, v2_results):
    """Compare V1 vs V2 results."""
    print("\n\n")
    print("="*80)
    print("V1 vs V2 Comparison")
    print("="*80)

    print(f"\n{'Question':<50} {'V1 Time':<12} {'V2 Time':<12} {'Diff':<10}")
    print("-"*80)

    total_v1_time = 0
    total_v2_time = 0

    for v1, v2 in zip(v1_results, v2_results):
        q = v1["question"][:47] + "..." if len(v1["question"]) > 50 else v1["question"]
        diff = v2["time"] - v1["time"]
        diff_str = f"+{diff:.1f}s" if diff > 0 else f"{diff:.1f}s"

        print(f"{q:<50} {v1['time']:>8.2f}s    {v2['time']:>8.2f}s    {diff_str:>8}")

        total_v1_time += v1["time"]
        total_v2_time += v2["time"]

    print("-"*80)
    print(f"{'AVERAGE':<50} {total_v1_time/len(v1_results):>8.2f}s    {total_v2_time/len(v2_results):>8.2f}s    {(total_v2_time-total_v1_time)/len(v1_results):+>8.1f}s")

    print("\n\nSummary:")
    print(f"  V1 average time: {total_v1_time/len(v1_results):.2f}s")
    print(f"  V2 average time: {total_v2_time/len(v2_results):.2f}s")
    print(f"  Time difference: {(total_v2_time-total_v1_time)/len(v1_results):+.2f}s ({((total_v2_time/total_v1_time - 1) * 100):+.1f}%)")

    print("\n\nV2 Retrieval Statistics:")
    for v2 in v2_results:
        if "debug" in v2 and v2["debug"]:
            debug = v2["debug"]
            print(f"\n  {v2['question'][:40]}...")
            print(f"    Expanded queries: {len(debug.get('expanded_queries', []))}")
            print(f"    Initial retrieval: {debug.get('initial_retrieval_count', 0)} docs")
            print(f"    After reranking: {debug.get('reranked_count', 0)} docs")
            print(f"    After filtering: {debug.get('filtered_count', 0)} docs")
            print(f"    Final: {debug.get('final_count', 0)} docs")


def main():
    """Run comparison test."""
    print("BioMoQA RAG V1 vs V2 Comparison Test")
    print("Testing on 5 sample questions")
    print()

    # Test V1
    print("Step 1/3: Testing V1 (baseline)")
    v1_results = test_v1()

    # Clean up GPU memory
    import torch
    torch.cuda.empty_cache()
    time.sleep(5)

    # Test V2
    print("\n\nStep 2/3: Testing V2 (enhanced)")
    v2_results = test_v2()

    # Compare
    print("\n\nStep 3/3: Comparing results")
    compare_results(v1_results, v2_results)

    print("\n\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Evaluate on full 120 QA pairs")
    print("  2. Measure ROUGE scores (V1 vs V2)")
    print("  3. Deploy V2 if improvements confirmed")


if __name__ == "__main__":
    main()
