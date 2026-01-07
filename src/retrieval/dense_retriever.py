"""
Dense retrieval using embeddings and FAISS vector similarity search.
Implements semantic search to complement BM25 keyword search.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import pickle


@dataclass
class Document:
    """Document with embedding"""
    pmcid: str
    title: str
    abstract: str
    embedding: Optional[np.ndarray] = None


class DenseRetriever:
    """
    Dense retrieval using embeddings and FAISS.

    Uses biomedical sentence transformer to encode documents and queries,
    then retrieves top-k most similar documents using cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        documents_path: Optional[str] = None
    ):
        """
        Args:
            model_name: Sentence transformer model
            index_path: Path to saved FAISS index
            documents_path: Path to saved documents
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

        if index_path and documents_path:
            self.load(index_path, documents_path)

    def build_index(self, documents: List[Document]):
        """
        Build FAISS index from documents.

        Args:
            documents: List of documents to index
        """
        print(f"Building dense index from {len(documents)} documents...")

        # Encode all documents
        texts = [f"{doc.title}. {doc.abstract}" for doc in documents]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        self.index.add(embeddings)

        # Store documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        self.documents = documents

        print(f"✓ Index built with {len(self.documents)} documents")

    def retrieve(self, query: str, top_k: int = 20) -> List[Document]:
        """
        Retrieve top-k most similar documents for query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of top-k documents ranked by similarity
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return documents with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc.score = float(score)
                results.append(doc)

        return results

    def save(self, index_path: str, documents_path: str):
        """Save index and documents to disk"""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save documents (without embeddings to save space)
        docs_to_save = []
        for doc in self.documents:
            doc_dict = {
                'pmcid': doc.pmcid,
                'title': doc.title,
                'abstract': doc.abstract
            }
            docs_to_save.append(doc_dict)

        with open(documents_path, 'wb') as f:
            pickle.dump(docs_to_save, f)

        print(f"✓ Saved index to {index_path}")
        print(f"✓ Saved documents to {documents_path}")

    def load(self, index_path: str, documents_path: str):
        """Load index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load documents
        with open(documents_path, 'rb') as f:
            docs_dict = pickle.load(f)

        self.documents = [
            Document(
                pmcid=d['pmcid'],
                title=d['title'],
                abstract=d['abstract']
            ) for d in docs_dict
        ]

        print(f"✓ Loaded index from {index_path}")
        print(f"✓ Loaded {len(self.documents)} documents from {documents_path}")


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (SIBILS) and dense retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
    """

    def __init__(
        self,
        sibils_retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
        k: int = 60
    ):
        """
        Args:
            sibils_retriever: SIBILS API retriever (BM25)
            dense_retriever: Dense vector retriever
            alpha: Weight for dense retrieval (0=BM25 only, 1=dense only)
            k: RRF constant (default 60 as in original paper)
        """
        self.sibils = sibils_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.k = k

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Document],
        dense_results: List[Document]
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) for each ranking
        """
        # Build RRF scores
        rrf_scores = {}

        # Add BM25 scores (weighted by 1 - alpha)
        for rank, doc in enumerate(bm25_results, start=1):
            key = doc.pmcid
            score = (1 - self.alpha) / (self.k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0) + score
            if key not in {d.pmcid: d for d in dense_results}:
                rrf_scores[f"{key}_doc"] = doc  # Store document reference

        # Add dense scores (weighted by alpha)
        for rank, doc in enumerate(dense_results, start=1):
            key = doc.pmcid
            score = self.alpha / (self.k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0) + score
            if key not in {d.pmcid: d for d in bm25_results}:
                rrf_scores[f"{key}_doc"] = doc

        # Combine documents from both sources
        all_docs = {d.pmcid: d for d in bm25_results + dense_results}

        # Sort by RRF score
        sorted_pmcids = sorted(
            [k for k in rrf_scores.keys() if not k.endswith('_doc')],
            key=lambda k: rrf_scores[k],
            reverse=True
        )

        # Return ranked documents
        results = []
        for pmcid in sorted_pmcids:
            if pmcid in all_docs:
                doc = all_docs[pmcid]
                doc.rrf_score = rrf_scores[pmcid]
                results.append(doc)

        return results

    def retrieve(self, query: str, n: int = 50, top_k: int = 20) -> List[Document]:
        """
        Retrieve documents using hybrid approach.

        Args:
            query: Search query
            n: Number of documents to retrieve from each source
            top_k: Final number of documents to return

        Returns:
            Top-k documents ranked by RRF score
        """
        # Retrieve from both sources
        bm25_results = self.sibils.retrieve(query, n=n)
        dense_results = self.dense.retrieve(query, top_k=n)

        # Combine using RRF
        hybrid_results = self.reciprocal_rank_fusion(bm25_results, dense_results)

        return hybrid_results[:top_k]
