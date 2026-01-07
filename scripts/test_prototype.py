#!/usr/bin/env python3
"""
Quick test script for BioMoQA-Ragnarok prototype

Tests the retrieval module without requiring the full LLM.
"""

import sys
import json
from src.retrieval import SIBILSRetriever


def test_retrieval():
    """Test SIBILS retrieval on sample questions."""
    print("=" * 80)
    print("BioMoQA-Ragnarok: Retrieval Module Test")
    print("=" * 80)
    print()

    # Initialize retriever
    retriever = SIBILSRetriever(collection="pmc", default_n=20)

    # Test questions
    questions = [
        "What is the host of Plasmodium falciparum?",
        "What parasites affect chickpea crops?",
        "What is the role of Anopheles mosquitoes in malaria transmission?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 80)

        try:
            # Retrieve documents
            documents = retriever.retrieve(question, n=5)

            print(f"Retrieved {len(documents)} documents:\n")

            for j, doc in enumerate(documents, 1):
                print(f"[{j}] {doc.doc_id} (score: {doc.score:.2f})")
                print(f"    Title: {doc.title}")
                print(f"    PMCID: {doc.pmcid or 'N/A'}")
                print(f"    Abstract: {doc.abstract[:150]}...")
                print()

        except Exception as e:
            print(f"ERROR: {str(e)}")
            return False

    print("=" * 80)
    print("✓ Retrieval test completed successfully!")
    print("=" * 80)
    return True


def test_full_pipeline():
    """Test full RAG pipeline (requires LLM)."""
    print("\n" + "=" * 80)
    print("BioMoQA-Ragnarok: Full Pipeline Test")
    print("=" * 80)
    print()

    print("WARNING: This will download ~8GB Llama model on first run!")
    response = input("Continue? [y/N]: ")

    if response.lower() != 'y':
        print("Skipping full pipeline test.")
        return True

    from src.pipeline import RAGPipeline, RAGConfig

    # Create minimal config
    config = RAGConfig(
        retrieval_n=20,
        rerank_n=5,
        load_in_4bit=True,
    )

    pipeline = RAGPipeline(config)

    # Test question
    question = "What is the host of Plasmodium falciparum?"

    result = pipeline.run(question, topic_id="TEST001")

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2))

    return True


if __name__ == "__main__":
    # Run retrieval test (always)
    success = test_retrieval()

    if not success:
        sys.exit(1)

    # Ask if user wants to test full pipeline
    print("\nRetrieval test passed!")
    print("\nWould you like to test the full pipeline (with LLM generation)?")
    response = input("This requires downloading Llama 3.1 8B (~8GB). Continue? [y/N]: ")

    if response.lower() == 'y':
        test_full_pipeline()
    else:
        print("\nSkipping full pipeline test.")
        print("To test later, run: python test_prototype.py --full")

    print("\n✓ All tests completed!")
