#!/usr/bin/env python3
"""
Debug script to test ES query mode with SIBILS API.
"""

import sys
sys.path.insert(0, '/home/egaillac/BioMoQA-RAG')

from src.retrieval.sibils_retriever import SIBILSRetriever
from src.retrieval.query_parser import SIBILSQueryParser
import json

def test_es_query():
    """Test ES query mode with debug output."""

    print("="*80)
    print("Testing ES Query Mode with SIBILS API")
    print("="*80)

    # Test question
    question = "What causes malaria?"
    print(f"\nQuestion: {question}\n")

    # First, let's see what the parser returns
    print("\n--- STEP 1: Parse Query ---")
    parser = SIBILSQueryParser(collection="pmc")
    parsed = parser.parse(question, include_es_query=True)

    print(f"Success: {parsed.success}")
    print(f"Normalized query: {parsed.normalized_query}")
    print(f"Text parts: {parsed.text_parts}")

    if parsed.es_query:
        print(f"\nES Query structure:")
        print(json.dumps(parsed.es_query, indent=2))

    # Now test retrieval with ES query
    print("\n\n--- STEP 2: Test Retrieval with ES Query ---")
    retriever = SIBILSRetriever(use_query_parser=True, use_es_query=True)
    docs = retriever.retrieve(question, n=10)

    print(f"\n\n--- RESULT ---")
    print(f"Retrieved: {len(docs)} documents")

    if docs:
        print("\nTop 3 documents:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"{i}. [{doc.pmcid or doc.doc_id}] (score: {doc.score:.2f})")
            print(f"   Title: {doc.title[:80]}...")
            print()

    # Compare with keywords mode
    print("\n\n--- STEP 3: Compare with Keywords Mode ---")
    retriever_keywords = SIBILSRetriever(use_query_parser=True, use_es_query=False)
    docs_keywords = retriever_keywords.retrieve(question, n=10)
    print(f"Keywords mode retrieved: {len(docs_keywords)} documents")

    if docs_keywords:
        print("\nTop 3 documents:")
        for i, doc in enumerate(docs_keywords[:3], 1):
            print(f"{i}. [{doc.pmcid or doc.doc_id}] (score: {doc.score:.2f})")
            print(f"   Title: {doc.title[:80]}...")
            print()

if __name__ == "__main__":
    test_es_query()
