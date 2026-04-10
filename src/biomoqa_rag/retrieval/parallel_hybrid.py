"""
Parallel hybrid retrieval for maximum speed.
Runs BM25 and Dense retrieval in parallel threads.
"""

from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ParallelHybridRetriever:
    """
    Hybrid retriever with parallel execution for speed.

    Runs BM25 (SIBILS) and dense (FAISS) retrieval in parallel,
    then combines results with RRF.
    """

    def __init__(
        self,
        sibils_retriever,
        dense_retriever,
        alpha: float = 0.5,
        k: int = 60,
        timeout: float = 10.0
    ):
        """
        Args:
            sibils_retriever: SIBILS API retriever
            dense_retriever: Dense FAISS retriever
            alpha: Weight for dense (0=BM25 only, 1=dense only)
            k: RRF constant
            timeout: Max wait time for retrievers
        """
        self.sibils = sibils_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.k = k
        self.timeout = timeout

    def reciprocal_rank_fusion(self, bm25_results, dense_results, collection=None):
        """Combine results using RRF"""
        rrf_scores = {}
        all_docs = {}

        # Add BM25 scores
        for rank, doc in enumerate(bm25_results, start=1):
            key = getattr(doc, 'doc_id', None) or getattr(doc, 'pmcid', None)
            score = (1 - self.alpha) / (self.k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0) + score
            all_docs[key] = doc

        # Normalise collection filter to a set of strings
        allowed = None
        if collection is not None:
            allowed = set(collection) if isinstance(collection, list) else {collection}

        # Add dense scores — skip FAISS docs not in the requested collection
        for rank, doc in enumerate(dense_results, start=1):
            if allowed and getattr(doc, 'source', None) not in allowed:
                continue
            key = getattr(doc, 'doc_id', None) or getattr(doc, 'pmcid', None)
            score = self.alpha / (self.k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0) + score
            all_docs[key] = doc

        # Sort by RRF score
        sorted_keys = sorted(
            rrf_scores.keys(),
            key=lambda k: rrf_scores[k],
            reverse=True
        )

        # Return ranked documents
        results = []
        for key in sorted_keys:
            doc = all_docs[key]
            doc.rrf_score = rrf_scores[key]
            results.append(doc)

        return results

    def retrieve(self, query: str, n: int = 20, top_k: int = 20, collection: Optional[str | list[str]] = None):
        """
        Retrieve documents using parallel hybrid approach.

        Args:
            query: Search query
            n: Number of documents from each source
            top_k: Final number to return
            collection: Override SIBILS collection(s)

        Returns:
            Top-k documents ranked by RRF
        """
        bm25_results = []
        dense_results = []
        errors = []

        # Run both retrievers in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_bm25 = executor.submit(self.sibils.retrieve, query, n=n, collection=collection)
            future_dense = executor.submit(self.dense.retrieve, query, n)

            # Collect results
            try:
                for future in as_completed([future_bm25, future_dense], timeout=self.timeout):
                    try:
                        if future == future_bm25:
                            bm25_results = future.result()
                        else:
                            dense_results = future.result()
                    except Exception as e:
                        errors.append(str(e))
            except TimeoutError:
                # One retriever timed out — use whatever finished
                if future_bm25.done():
                    try:
                        bm25_results = future_bm25.result()
                    except Exception as e:
                        errors.append(str(e))
                if future_dense.done():
                    try:
                        dense_results = future_dense.result()
                    except Exception as e:
                        errors.append(str(e))

        # Fallback if one method fails
        if not bm25_results and not dense_results:
            raise RuntimeError(f"Both retrievers failed: {errors}")

        if not bm25_results:
            # Dense only
            return dense_results[:top_k]

        if not dense_results:
            # BM25 only
            return bm25_results[:top_k]

        # Combine with RRF
        hybrid_results = self.reciprocal_rank_fusion(bm25_results, dense_results, collection=collection)

        return hybrid_results[:top_k]


class SmartHybridRetriever:
    """
    Smart hybrid retriever that adapts based on query characteristics.

    Uses heuristics to decide whether to use BM25, dense, or hybrid:
    - Short queries with specific terms -> BM25
    - Longer semantic queries -> Dense
    - Medical/technical queries -> Hybrid
    """

    def __init__(
        self,
        sibils_retriever,
        dense_retriever,
        alpha: float = 0.5,
        k: int = 60
    ):
        self.sibils = sibils_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.k = k
        self.parallel = ParallelHybridRetriever(
            sibils_retriever,
            dense_retriever,
            alpha,
            k
        )

        # Keywords that suggest technical/medical queries
        self.technical_keywords = {
            'ag1', 'ia', 'pmcid', 'pmid', 'gene', 'protein',
            'enzyme', 'antibody', 'receptor', 'pathway',
            'mutation', 'variant', 'strain', 'species'
        }

    def should_use_hybrid(self, query: str) -> str:
        """
        Decide which retrieval method to use.

        Returns: 'bm25', 'dense', or 'hybrid'
        """
        query_lower = query.lower()
        words = query_lower.split()

        # Check for technical terms
        has_technical = any(kw in query_lower for kw in self.technical_keywords)

        # Short queries with technical terms -> BM25 (fastest)
        if len(words) <= 5 and has_technical:
            return 'bm25'

        # Long semantic queries -> Dense (very fast)
        if len(words) > 10 and not has_technical:
            return 'dense'

        # Everything else -> Hybrid (best quality)
        return 'hybrid'

    def retrieve(self, query: str, n: int = 20, top_k: int = 20, collection: Optional[str | list[str]] = None):
        """
        Smart retrieval that chooses method based on query.

        Args:
            query: Search query
            n: Number of documents from each source
            top_k: Final number to return
            collection: Override SIBILS collection(s)

        Returns:
            Top-k documents
        """
        # When a specific collection is requested, always use BM25 or hybrid —
        # never dense-only. FAISS has limited per-collection coverage and its
        # source filtering is unreliable, causing the same docs to bleed across
        # collections. SIBILS BM25 knows the collection boundaries authoritatively.
        if collection is not None:
            return self.parallel.retrieve(query, n=n, top_k=top_k, collection=collection)

        method = self.should_use_hybrid(query)

        if method == 'bm25':
            # Fast path: BM25 only
            return self.sibils.retrieve(query, n=top_k, collection=collection)

        elif method == 'dense':
            # Fastest path: Dense only (no collection filter needed here — collection is None)
            return self.dense.retrieve(query, top_k=top_k)

        else:
            # Best quality: Parallel hybrid
            return self.parallel.retrieve(query, n=n, top_k=top_k, collection=collection)
