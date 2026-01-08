#!/usr/bin/env python3
"""
Test Phase 2 hybrid retrieval performance and speed.
Compare different configurations to optimize speed.
"""

import time
from src.retrieval.sibils_retriever import SIBILSRetriever
from src.retrieval.dense_retriever import DenseRetriever, HybridRetriever

def test_retrieval_speed():
    """Test retrieval speed for different approaches"""

    print("="*80)
    print("Phase 2 Retrieval Speed Test")
    print("="*80)
    print()

    # Initialize retrievers
    print("Initializing retrievers...")
    sibils = SIBILSRetriever()

    dense = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    dense.load("data/faiss_index.bin", "data/documents.pkl")
    print(f"✓ Loaded {len(dense.documents)} documents in dense index")
    print()

    # Test questions
    test_questions = [
        "What causes malaria?",
        "What is AG1-IA?",
        "What are the symptoms of diabetes?",
        "How does the immune system fight viruses?",
        "What is the role of hemoglobin?"
    ]

    results = {}

    # Test 1: BM25 only (baseline)
    print("1. BM25 ONLY (SIBILS) - Baseline")
    print("-" * 40)
    times_bm25 = []
    for q in test_questions:
        start = time.time()
        docs = sibils.retrieve(q, n=20)
        elapsed = time.time() - start
        times_bm25.append(elapsed)
        print(f"  {q[:50]:50s} {elapsed:.3f}s ({len(docs)} docs)")

    avg_bm25 = sum(times_bm25) / len(times_bm25)
    print(f"\nAverage: {avg_bm25:.3f}s")
    results['bm25_only'] = avg_bm25
    print()

    # Test 2: Dense only
    print("2. DENSE ONLY (FAISS)")
    print("-" * 40)
    times_dense = []
    for q in test_questions:
        start = time.time()
        docs = dense.retrieve(q, top_k=20)
        elapsed = time.time() - start
        times_dense.append(elapsed)
        print(f"  {q[:50]:50s} {elapsed:.3f}s ({len(docs)} docs)")

    avg_dense = sum(times_dense) / len(times_dense)
    print(f"\nAverage: {avg_dense:.3f}s")
    results['dense_only'] = avg_dense
    print()

    # Test 3: Hybrid (50 from each)
    print("3. HYBRID (50 from each, alpha=0.5)")
    print("-" * 40)
    hybrid = HybridRetriever(sibils, dense, alpha=0.5, k=60)
    times_hybrid = []
    for q in test_questions:
        start = time.time()
        docs = hybrid.retrieve(q, n=50, top_k=20)
        elapsed = time.time() - start
        times_hybrid.append(elapsed)
        print(f"  {q[:50]:50s} {elapsed:.3f}s ({len(docs)} docs)")

    avg_hybrid = sum(times_hybrid) / len(times_hybrid)
    print(f"\nAverage: {avg_hybrid:.3f}s")
    results['hybrid_50_50'] = avg_hybrid
    print()

    # Test 4: Hybrid optimized (30 from each)
    print("4. HYBRID OPTIMIZED (30 from each, alpha=0.5)")
    print("-" * 40)
    times_hybrid_opt = []
    for q in test_questions:
        start = time.time()
        docs = hybrid.retrieve(q, n=30, top_k=20)
        elapsed = time.time() - start
        times_hybrid_opt.append(elapsed)
        print(f"  {q[:50]:50s} {elapsed:.3f}s ({len(docs)} docs)")

    avg_hybrid_opt = sum(times_hybrid_opt) / len(times_hybrid_opt)
    print(f"\nAverage: {avg_hybrid_opt:.3f}s")
    results['hybrid_30_30'] = avg_hybrid_opt
    print()

    # Test 5: Hybrid fast (20 from each)
    print("5. HYBRID FAST (20 from each, alpha=0.5)")
    print("-" * 40)
    times_hybrid_fast = []
    for q in test_questions:
        start = time.time()
        docs = hybrid.retrieve(q, n=20, top_k=20)
        elapsed = time.time() - start
        times_hybrid_fast.append(elapsed)
        print(f"  {q[:50]:50s} {elapsed:.3f}s ({len(docs)} docs)")

    avg_hybrid_fast = sum(times_hybrid_fast) / len(times_hybrid_fast)
    print(f"\nAverage: {avg_hybrid_fast:.3f}s")
    results['hybrid_20_20'] = avg_hybrid_fast
    print()

    # Summary
    print("="*80)
    print("SPEED COMPARISON SUMMARY")
    print("="*80)
    print()
    print(f"BM25 only (baseline):          {results['bm25_only']:.3f}s")
    print(f"Dense only:                    {results['dense_only']:.3f}s  ({results['dense_only']/results['bm25_only']*100-100:+.1f}%)")
    print(f"Hybrid (50 from each):         {results['hybrid_50_50']:.3f}s  ({results['hybrid_50_50']/results['bm25_only']*100-100:+.1f}%)")
    print(f"Hybrid optimized (30 each):    {results['hybrid_30_30']:.3f}s  ({results['hybrid_30_30']/results['bm25_only']*100-100:+.1f}%)")
    print(f"Hybrid fast (20 each):         {results['hybrid_20_20']:.3f}s  ({results['hybrid_20_20']/results['bm25_only']*100-100:+.1f}%)")
    print()

    # Recommendation
    print("RECOMMENDATIONS:")
    print("-" * 40)

    if results['hybrid_20_20'] < results['bm25_only'] * 1.5:
        print("✓ HYBRID FAST (20 from each) is recommended")
        print(f"  - Only {results['hybrid_20_20']/results['bm25_only']*100-100:+.1f}% slower than BM25")
        print("  - Provides semantic search benefits")
        print("  - Minimal overhead for quality gains")
    elif results['hybrid_30_30'] < results['bm25_only'] * 1.8:
        print("✓ HYBRID OPTIMIZED (30 from each) is recommended")
        print(f"  - Only {results['hybrid_30_30']/results['bm25_only']*100-100:+.1f}% slower than BM25")
        print("  - Better coverage than 20x20")
    else:
        print("⚠ Hybrid retrieval adds significant overhead")
        print("  Consider using dense retrieval in parallel")
        print("  Or implement caching strategies")

    print()
    print("SPEED OPTIMIZATION IDEAS:")
    print("-" * 40)
    print("1. Parallel retrieval: Run BM25 and Dense in parallel threads")
    print("2. Smaller index: Use only most relevant documents (~500-1000)")
    print("3. Faster embedding model: Use smaller sentence-transformer")
    print("4. Query caching: Cache frequent queries")
    print("5. Skip query expansion for simple queries")

    return results

if __name__ == "__main__":
    test_retrieval_speed()
