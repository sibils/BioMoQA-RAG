#!/usr/bin/env python3
"""
Final test of ES query mode in the full RAG pipeline.
"""

import sys
sys.path.insert(0, '/home/egaillac/BioMoQA-RAG')

from src.retrieval.sibils_retriever import SIBILSRetriever

def test_final():
    """Test ES query mode with default settings."""

    print("="*80)
    print("Final Test: ES Query Mode (Default Settings)")
    print("="*80)

    # Test questions
    questions = [
        "What causes malaria?",
        "What is the host of Plasmodium falciparum?",
        "How does the immune system respond to viral infections?"
    ]

    # Initialize retriever with default settings (ES query should be enabled)
    retriever = SIBILSRetriever()
    print(f"\nRetriever settings:")
    print(f"  - use_query_parser: {retriever.use_query_parser}")
    print(f"  - use_es_query: {retriever.use_es_query}")

    for question in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print("="*80)

        # Retrieve documents
        docs = retriever.retrieve(question, n=5)

        print(f"\nRetrieved: {len(docs)} documents")

        if docs:
            print("\nTop 3 results:")
            for i, doc in enumerate(docs[:3], 1):
                print(f"{i}. [{doc.pmcid or doc.doc_id}] (score: {doc.score:.2f})")
                print(f"   {doc.title[:100]}...")
        else:
            print("WARNING: No documents retrieved!")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)

if __name__ == "__main__":
    test_final()
