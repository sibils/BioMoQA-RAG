#!/usr/bin/env python3
"""
Simple non-interactive test of BioMoQA-Ragnarok

This script tests the full RAG pipeline with a single question.
Perfect for quickly validating the system works end-to-end.
"""

import json
import sys
from src.pipeline import RAGPipeline, RAGConfig


def main():
    print("="*80)
    print("BioMoQA-Ragnarok: Simple RAG Test")
    print("="*80)
    print()

    # Configuration
    config = RAGConfig(
        retrieval_n=50,          # Retrieve 50 documents
        rerank_n=10,             # Keep top-10 after reranking
        model_name="Qwen/Qwen2.5-7B-Instruct",  # Fully open model, no gating
        load_in_4bit=True,       # 4-bit quantization (saves memory)
        max_new_tokens=512,      # Max answer length
        temperature=0.7,         # Sampling temperature
    )

    print("Initializing pipeline...")
    pipeline = RAGPipeline(config)
    print()

    # Test question
    question = "What is the host of Plasmodium falciparum?"
    print(f"Question: {question}\n")

    # Run pipeline
    result = pipeline.run(
        question=question,
        topic_id="TEST001",
        return_documents=True,
    )

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print()

    print(f"Pipeline Time: {result['pipeline_time']:.2f}s")
    print(f"Documents Retrieved: {result['num_retrieved']}")
    print()

    print("Answer:")
    print("-" * 80)
    if result['answer']:
        for sent in result['answer']:
            citations = ", ".join([f"[{c}]" for c in sent['citations']])
            print(f"  {sent['text']} {citations}")
    else:
        print(result.get('raw_answer', 'No answer generated'))
    print()

    print("Top Retrieved Documents:")
    print("-" * 80)
    if 'documents' in result:
        for i, doc in enumerate(result['documents'][:5], 1):
            print(f"{i}. [{doc['id']}] {doc['title'][:80]}...")
            print(f"   Score: {doc['score']:.2f}")
    print()

    # Save results
    output_file = "results/simple_test_output.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Full output saved to: {output_file}")

    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
