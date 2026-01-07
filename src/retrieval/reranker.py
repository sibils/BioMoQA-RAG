"""
Cross-Encoder Reranker for Semantic Relevance Scoring

Uses a cross-encoder model to rerank retrieved documents based on
semantic relevance to the question.
"""

from typing import List
from sentence_transformers import CrossEncoder
import numpy as np

from . import Document


class SemanticReranker:
    """
    Rerank documents using cross-encoder for semantic relevance.

    Cross-encoders score (question, document) pairs directly,
    providing better relevance scores than bi-encoders.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize reranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model name for cross-encoder
                       Default: ms-marco-MiniLM (fast, good for biomedical)
                       Alternative: cross-encoder/ms-marco-MedMarco-electra-base (slower, better)
        """
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print("✓ Reranker loaded")

    def rerank(
        self,
        question: str,
        documents: List[Document],
        top_k: int = 20
    ) -> List[Document]:
        """
        Rerank documents by semantic relevance to question.

        Args:
            question: The query question
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            List of top-k documents sorted by relevance
        """
        if len(documents) == 0:
            return []

        if len(documents) <= top_k:
            # If we have fewer docs than requested, just score and sort
            return self._score_and_sort(question, documents)

        # Create (question, document) pairs for cross-encoder
        pairs = []
        for doc in documents:
            # Combine title and abstract for better matching
            doc_text = f"{doc.title}. {doc.abstract}"
            pairs.append([question, doc_text])

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort documents by score (descending)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Return top-k
        reranked = [doc for score, doc in scored_docs[:top_k]]

        return reranked

    def _score_and_sort(self, question: str, documents: List[Document]) -> List[Document]:
        """Score and sort documents when count <= top_k."""
        pairs = [[question, f"{doc.title}. {doc.abstract}"] for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs]


class HybridReranker:
    """
    Combine original retrieval scores with cross-encoder scores.

    Uses weighted combination of BM25 scores and semantic scores
    for more robust ranking.
    """

    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        """
        Initialize hybrid reranker.

        Args:
            model_name: Cross-encoder model
            bm25_weight: Weight for original BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
        """
        self.model = CrossEncoder(model_name)
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Normalize weights
        total = bm25_weight + semantic_weight
        self.bm25_weight /= total
        self.semantic_weight /= total

    def rerank(
        self,
        question: str,
        documents: List[Document],
        top_k: int = 20
    ) -> List[Document]:
        """Rerank with hybrid scoring."""
        if len(documents) == 0:
            return []

        # Get semantic scores
        pairs = [[question, f"{doc.title}. {doc.abstract}"] for doc in documents]
        semantic_scores = self.model.predict(pairs, show_progress_bar=False)

        # Normalize scores to 0-1
        sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
        if sem_max > sem_min:
            semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min)

        # Get original BM25 scores (if available)
        bm25_scores = np.array([doc.score if hasattr(doc, 'score') and doc.score else 0.5
                                for doc in documents])
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)

        # Combine scores
        final_scores = (self.bm25_weight * bm25_scores +
                       self.semantic_weight * semantic_scores)

        # Sort and return top-k
        scored_docs = list(zip(final_scores, documents))
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        return [doc for score, doc in scored_docs[:top_k]]
