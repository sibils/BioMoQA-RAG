"""
SIBILS API Retriever for Biomedical Documents

This module implements the retrieval stage (R) of the Ragnarok RAG pipeline.
It queries the SIBILS biodiversity/biomedical API to retrieve relevant documents.
"""

import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class Document:
    """Represents a retrieved document with metadata."""
    doc_id: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    score: float = 0.0
    source: str = "sibils"
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None

    def get_text(self, max_length: Optional[int] = None) -> str:
        """Get document text (prioritize full_text, fallback to abstract)."""
        text = self.full_text if self.full_text else f"{self.title}\n{self.abstract}"
        if max_length:
            return text[:max_length]
        return text


class SIBILSRetriever:
    """
    Retrieval module using SIBILS API.

    SIBILS provides access to:
    - PMC (PubMed Central) full-text articles
    - PubMed abstracts
    - Plazi biodiversity treatments
    - Supplementary data

    Based on Ragnarok framework retrieval stage.
    """

    def __init__(
        self,
        api_url: str = "https://biodiversitypmc.sibils.org/api/search",
        collection: str = "pmc",
        default_n: int = 100,
        timeout: int = 30,
    ):
        """
        Initialize SIBILS retriever.

        Args:
            api_url: SIBILS API endpoint
            collection: Collection to search ("pmc", "medline", "plazi", "suppdata")
            default_n: Default number of documents to retrieve
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.collection = collection
        self.default_n = default_n
        self.timeout = timeout

    def retrieve(
        self,
        question: str,
        n: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve documents from SIBILS API.

        Args:
            question: User query/question
            n: Number of documents to retrieve (default: self.default_n)
            collection: Override default collection

        Returns:
            List of Document objects sorted by relevance
        """
        n = n or self.default_n
        collection = collection or self.collection

        params = {
            "q": question,
            "col": collection,
            "n": n,
        }

        try:
            response = requests.get(
                self.api_url,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                error = data.get("error", "Unknown error")
                raise Exception(f"SIBILS API error: {error}")

            # Extract documents from response
            hits = data.get("elastic_output", {}).get("hits", {}).get("hits", [])
            documents = []

            for i, hit in enumerate(hits):
                source = hit.get("_source", {})

                doc = Document(
                    doc_id=hit.get("_id", f"doc_{i}"),
                    title=source.get("title", ""),
                    abstract=source.get("abstract", ""),
                    full_text=source.get("full_text"),
                    score=hit.get("_score", 0.0),
                    pmid=source.get("pmid"),
                    pmcid=source.get("pmcid"),
                    doi=source.get("doi"),
                )
                documents.append(doc)

            return documents

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to query SIBILS API: {str(e)}")

    def retrieve_batch(
        self,
        questions: List[str],
        n: Optional[int] = None,
        delay: float = 0.5,
    ) -> List[List[Document]]:
        """
        Retrieve documents for multiple questions (with rate limiting).

        Args:
            questions: List of questions
            n: Number of documents per question
            delay: Delay between requests (seconds)

        Returns:
            List of document lists (one per question)
        """
        results = []
        for question in questions:
            docs = self.retrieve(question, n=n)
            results.append(docs)
            time.sleep(delay)  # Rate limiting
        return results


def main():
    """Example usage of SIBILS retriever."""
    retriever = SIBILSRetriever()

    # Example question
    question = "What is the host of Plasmodium falciparum?"
    print(f"Question: {question}\n")

    # Retrieve documents
    documents = retriever.retrieve(question, n=5)

    print(f"Retrieved {len(documents)} documents:\n")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. [{doc.doc_id}] (score: {doc.score:.2f})")
        print(f"   Title: {doc.title}")
        print(f"   Abstract: {doc.abstract[:200]}...")
        print()


if __name__ == "__main__":
    main()
