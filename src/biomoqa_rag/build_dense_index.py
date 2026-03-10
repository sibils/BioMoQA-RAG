#!/usr/bin/env python3
"""
Build FAISS dense index from SIBILS documents.

This script:
1. Queries SIBILS API with general biomedical terms to collect a corpus
2. Builds embeddings for all documents
3. Creates FAISS index for fast similarity search
4. Saves index and documents to disk
"""

from .retrieval.sibils_retriever import SIBILSRetriever
from .retrieval.dense_retriever import DenseRetriever, Document
from pathlib import Path
from tqdm import tqdm
import time

SEED_QUERIES_FILE = Path.cwd() / "data" / "seed_queries.txt"


def load_seed_queries(path: Path = SEED_QUERIES_FILE):
    """Load seed queries from file, ignoring blank lines and comments (#)."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def build_biomedical_corpus():
    """
    Build a corpus by querying SIBILS with broad biomedical terms.

    Queries are loaded from data/seed_queries.txt so they can be
    edited without touching the code.
    """
    retriever = SIBILSRetriever()

    seed_queries = load_seed_queries()
    print(f"Loaded {len(seed_queries)} seed queries from {SEED_QUERIES_FILE}")

    print(f"Building corpus from {len(seed_queries)} seed queries...")
    print("This will take approximately 5-10 minutes...")
    print()

    all_documents = {}  # Use dict to deduplicate by PMCID

    for query in tqdm(seed_queries, desc="Querying SIBILS"):
        try:
            # Retrieve 100 documents per query
            results = retriever.retrieve(query, n=100)

            for doc in results:
                if doc.pmcid not in all_documents:
                    all_documents[doc.pmcid] = Document(
                        pmcid=doc.pmcid,
                        title=doc.title,
                        abstract=doc.abstract
                    )

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"Error with query '{query}': {e}")
            continue

    documents = list(all_documents.values())
    print(f"\n✓ Collected {len(documents)} unique documents")

    return documents


def main():
    print("="*80)
    print("Building Dense Index for BioMoQA RAG")
    print("="*80)
    print()

    # Step 1: Build corpus
    print("Step 1: Building biomedical corpus from SIBILS...")
    documents = build_biomedical_corpus()

    if len(documents) < 100:
        print(f"⚠ Warning: Only collected {len(documents)} documents")
        print("  This may not provide good coverage")

    # Step 2: Build FAISS index
    print()
    print("Step 2: Building FAISS index with embeddings...")
    print("Using model: sentence-transformers/all-MiniLM-L6-v2")
    print()

    dense_retriever = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    dense_retriever.build_index(documents)

    # Step 3: Save to disk
    print()
    print("Step 3: Saving index to disk...")
    index_path = "data/faiss_index.bin"
    docs_path = "data/documents.pkl"

    dense_retriever.save(index_path, docs_path)

    # Step 4: Test retrieval
    print()
    print("Step 4: Testing retrieval...")
    test_query = "What causes malaria?"
    results = dense_retriever.retrieve(test_query, top_k=5)

    print(f"\nTest query: '{test_query}'")
    print(f"Top 5 results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.title} (PMC{doc.pmcid})")
        print(f"   Score: {doc.score:.3f}")
        print(f"   {doc.abstract[:150]}...")

    print()
    print("="*80)
    print("✓ Dense index successfully built!")
    print("="*80)
    print()
    print(f"Index saved to: {index_path}")
    print(f"Documents saved to: {docs_path}")
    print(f"Total documents: {len(documents)}")
    print()
    print("You can now use hybrid retrieval in your pipeline:")
    print("  from src.biomoqa_rag.retrieval.dense_retriever import DenseRetriever, HybridRetriever")
    print(f"  dense = DenseRetriever()")
    print(f"  dense.load('{index_path}', '{docs_path}')")
    print(f"  hybrid = HybridRetriever(sibils_retriever, dense, alpha=0.5)")

if __name__ == "__main__":
    main()
