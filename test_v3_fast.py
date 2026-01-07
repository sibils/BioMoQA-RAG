#!/usr/bin/env python3
"""
Test V3.1 (ultra-fast) vs V3 and different quantization methods.
"""

import time
import argparse
from src.pipeline_vllm_v3_fast import UltraFastRAGPipeline, RAGConfigV3Fast


def test_v3_fast(quantization=None):
    """Test V3.1 with specified quantization"""

    print("="*80)
    print(f"Testing V3.1 Pipeline" + (f" with {quantization}" if quantization else ""))
    print("="*80)
    print()

    # Initialize pipeline
    config = RAGConfigV3Fast(
        retrieval_n=20,
        use_smart_retrieval=True,
        use_reranking=True,
        final_n=10,
        max_tokens=384,
        truncate_abstracts=True,
        max_abstract_length=800,
        quantization=quantization
    )

    pipeline = UltraFastRAGPipeline(config)

    # Test questions
    test_questions = [
        "What causes malaria?",
        "What is AG1-IA?",
        "What are the symptoms of diabetes?",
        "How does the immune system fight viruses?",
        "What is the role of hemoglobin?"
    ]

    print("\nRunning V3.1 pipeline on test questions...")
    print("="*80)
    print()

    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}/{len(test_questions)}: {question}")
        print("-" * 80)

        start = time.time()
        result = pipeline.run(question, debug=True)
        elapsed = time.time() - start

        results.append({
            'question': question,
            'time': elapsed,
            'num_docs': result['num_retrieved'],
            'answer_len': len(result['answer']),
            'debug': result.get('debug_info', {})
        })

        # Print results
        print(f"Time: {elapsed:.3f}s")
        print(f"Documents: {result['num_retrieved']}")
        print(f"Answer length: {len(result['answer'])} chars")

        if 'debug_info' in result:
            debug = result['debug_info']
            print(f"\nBreakdown:")
            if 'retrieval_time' in debug:
                print(f"  Retrieval: {debug['retrieval_time']:.3f}s")
            if 'rerank_time' in debug:
                print(f"  Reranking: {debug['rerank_time']:.3f}s")
            if 'filter_time' in debug:
                print(f"  Filtering: {debug['filter_time']:.3f}s")
            if 'generation_time' in debug:
                print(f"  Generation: {debug['generation_time']:.3f}s")

        print(f"\nAnswer: {result['answer'][:200]}...")
        print()

    # Summary
    print("="*80)
    print(f"V3.1 SUMMARY" + (f" ({quantization})" if quantization else ""))
    print("="*80)
    print()

    avg_time = sum(r['time'] for r in results) / len(results)
    avg_docs = sum(r['num_docs'] for r in results) / len(results)
    avg_answer_len = sum(r['answer_len'] for r in results) / len(results)

    # Calculate average breakdown
    avg_retrieval = sum(r['debug'].get('retrieval_time', 0) for r in results) / len(results)
    avg_rerank = sum(r['debug'].get('rerank_time', 0) for r in results) / len(results)
    avg_filter = sum(r['debug'].get('filter_time', 0) for r in results) / len(results)
    avg_generation = sum(r['debug'].get('generation_time', 0) for r in results) / len(results)

    print(f"Average pipeline time: {avg_time:.3f}s")
    print(f"Average documents: {avg_docs:.1f}")
    print(f"Average answer length: {avg_answer_len:.0f} chars")
    print()
    print("Time breakdown:")
    print(f"  Retrieval:  {avg_retrieval:.3f}s ({avg_retrieval/avg_time*100:.1f}%)")
    print(f"  Reranking:  {avg_rerank:.3f}s ({avg_rerank/avg_time*100:.1f}%)")
    print(f"  Filtering:  {avg_filter:.3f}s ({avg_filter/avg_time*100:.1f}%)")
    print(f"  Generation: {avg_generation:.3f}s ({avg_generation/avg_time*100:.1f}%)")
    print()

    # Compare with V3
    v3_time = 6.81
    v3_generation = 4.86
    improvement = (v3_time - avg_time) / v3_time * 100
    gen_improvement = (v3_generation - avg_generation) / v3_generation * 100

    print("="*80)
    print("V3 vs V3.1 COMPARISON")
    print("="*80)
    print()
    print(f"V3 total time:        {v3_time:.2f}s")
    print(f"V3.1 total time:      {avg_time:.2f}s")
    print(f"Total speedup:        {improvement:+.1f}%")
    print()
    print(f"V3 generation time:   {v3_generation:.2f}s")
    print(f"V3.1 generation time: {avg_generation:.2f}s")
    print(f"Generation speedup:   {gen_improvement:+.1f}%")
    print()

    if improvement > 30:
        print("🚀 EXCELLENT! V3.1 is significantly faster than V3")
    elif improvement > 20:
        print("✓ VERY GOOD! V3.1 provides substantial speed improvement")
    elif improvement > 10:
        print("✓ GOOD! V3.1 is noticeably faster than V3")
    elif improvement > 0:
        print("✓ V3.1 is faster than V3")
    else:
        print("⚠ V3.1 is not faster, check configuration")

    return {
        'avg_time': avg_time,
        'avg_generation': avg_generation,
        'improvement': improvement,
        'gen_improvement': gen_improvement
    }


def compare_quantizations():
    """Compare different quantization methods"""

    print("="*80)
    print("Comparing Quantization Methods")
    print("="*80)
    print()

    methods = [
        (None, "No quantization (baseline)"),
        ("fp8", "FP8 quantization"),
    ]

    all_results = {}

    for quant_method, description in methods:
        print(f"\nTesting: {description}")
        print("-" * 80)

        try:
            results = test_v3_fast(quantization=quant_method)
            all_results[quant_method or 'baseline'] = results
        except Exception as e:
            print(f"✗ Error with {quant_method}: {e}")
            continue

        print("\n" + "="*80 + "\n")

    # Summary comparison
    if len(all_results) > 1:
        print("="*80)
        print("QUANTIZATION COMPARISON SUMMARY")
        print("="*80)
        print()

        baseline = all_results.get('baseline', {})
        baseline_time = baseline.get('avg_time', 0)

        print(f"{'Method':<20} {'Total Time':<12} {'Speedup':<12} {'Gen Time':<12} {'Gen Speedup':<12}")
        print("-" * 80)

        for method, results in all_results.items():
            total_time = results['avg_time']
            gen_time = results['avg_generation']

            if baseline_time > 0 and method != 'baseline':
                speedup = (baseline_time - total_time) / baseline_time * 100
                speedup_str = f"{speedup:+.1f}%"
            else:
                speedup_str = "baseline"

            print(f"{method or 'baseline':<20} {total_time:.3f}s{'':<6} {speedup_str:<12} {gen_time:.3f}s{'':<6}")

        print()
        print("Recommendation:")
        if 'fp8' in all_results and baseline_time > 0:
            fp8_improvement = (baseline_time - all_results['fp8']['avg_time']) / baseline_time * 100
            if fp8_improvement > 20:
                print("  → Use FP8 quantization (significant speedup)")
            elif fp8_improvement > 10:
                print("  → Use FP8 quantization (good speedup)")
            else:
                print("  → FP8 provides modest improvement, evaluate quality tradeoff")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test V3.1 fast pipeline")
    parser.add_argument('--method', type=str, default=None,
                        help='Quantization method: fp8, awq, gptq, or None')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different quantization methods')

    args = parser.parse_args()

    if args.compare:
        compare_quantizations()
    else:
        test_v3_fast(quantization=args.method)
