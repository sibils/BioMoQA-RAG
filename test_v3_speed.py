#!/usr/bin/env python3
"""
Test V3 pipeline speed and compare with V2.
"""

import time
from src.pipeline_vllm_v3 import FastRAGPipelineV3, RAGConfigV3

def test_v3_speed():
    """Test V3 pipeline speed"""

    print("="*80)
    print("V3 Pipeline Speed Test")
    print("="*80)
    print()

    # Initialize V3 pipeline
    config = RAGConfigV3(
        retrieval_n=20,
        use_smart_retrieval=True,
        use_reranking=True,
        use_relevance_filter=True,
        final_n=15,
        max_tokens=512
    )

    pipeline = FastRAGPipelineV3(config)

    # Test questions
    test_questions = [
        "What causes malaria?",
        "What is AG1-IA?",
        "What are the symptoms of diabetes?",
        "How does the immune system fight viruses?",
        "What is the role of hemoglobin?"
    ]

    print("\nRunning V3 pipeline on test questions...")
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
    print("V3 SPEED SUMMARY")
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

    # Compare with V2 target
    v2_time = 11.19  # From evaluation
    improvement = (v2_time - avg_time) / v2_time * 100

    print("="*80)
    print("V2 vs V3 COMPARISON")
    print("="*80)
    print()
    print(f"V2 average time: {v2_time:.2f}s")
    print(f"V3 average time: {avg_time:.2f}s")
    print(f"Speed improvement: {improvement:+.1f}%")
    print()

    if improvement > 20:
        print("🚀 EXCELLENT! V3 is significantly faster than V2")
    elif improvement > 10:
        print("✓ GOOD! V3 provides meaningful speed improvement")
    elif improvement > 0:
        print("✓ V3 is faster than V2")
    else:
        print("⚠ V3 is slower than V2, needs optimization")

    print()
    print("Next steps:")
    print("  1. Run full 120 QA evaluation")
    print("  2. Compare quality metrics with V2")
    print("  3. Tune parameters for optimal speed/quality tradeoff")

if __name__ == "__main__":
    test_v3_speed()
