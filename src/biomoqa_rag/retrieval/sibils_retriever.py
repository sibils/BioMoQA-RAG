"""
SIBILS API Retriever for Biomedical Documents

This module implements the retrieval stage (R) of the Ragnarok RAG pipeline.
It queries the SIBILS biodiversity/biomedical API to retrieve relevant documents.
"""

import hashlib
import pickle
import shelve
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import requests

from .query_parser import SIBILSQueryParser, ParsedQuery


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
    - Medline (PubMed abstracts)
    - Plazi biodiversity treatments
    - PMC (PubMed Central) full-text articles
    - Supplementary data

    Based on Ragnarok framework retrieval stage.
    """

    # Valid collection names
    VALID_COLLECTIONS = {"medline", "plazi", "pmc"}

    def __init__(
        self,
        api_url: str = "https://biodiversitypmc.sibils.org/api/search",
        collection: str | list[str] = None,
        default_n: int = 100,
        timeout: int = 30,
        use_query_parser: bool = True,
        use_es_query: bool = True,  # Use full Elasticsearch query with concept annotations
        cache_dir: Optional[str] = "data/sibils_cache",
        cache_ttl: int = 604800,  # 7 days
    ):
        """
        Initialize SIBILS retriever.

        Args:
            api_url: SIBILS API endpoint
            collection: Collection(s) to search. A single string ("medline", "plazi",
                        "pmc") or a list of collections.
                        Defaults to ["medline", "plazi"].
            default_n: Default number of documents to retrieve
            timeout: Request timeout in seconds
            use_query_parser: Use SIBILS query parser to enhance queries
            use_es_query: Use Elasticsearch query (requires query parser)
            cache_dir: Directory for disk cache (None to disable)
            cache_ttl: Cache time-to-live in seconds (default 7 days)
        """
        self.api_url = api_url
        if collection is None:
            self.collection = ["medline", "plazi"]
        else:
            self.collection = collection
        self.default_n = default_n
        self.timeout = timeout
        self.use_query_parser = use_query_parser
        self.use_es_query = use_es_query and use_query_parser  # ES query requires parser

        # Initialize query parser if enabled (use first collection for ES query generation)
        if self.use_query_parser:
            first_col = self.collection[0] if isinstance(self.collection, list) else self.collection
            self.query_parser = SIBILSQueryParser(collection=first_col)
        else:
            self.query_parser = None

        # Disk cache
        self._cache_path: Optional[str] = None
        self._cache_ttl: int = cache_ttl
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._cache_path = str(Path(cache_dir) / "cache")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, question: str, collection: str, n: int) -> str:
        raw = f"{question.lower().strip()}|{collection}|{n}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, question: str, collection: str, n: int) -> Optional[List["Document"]]:
        if not self._cache_path:
            return None
        key = self._cache_key(question, collection, n)
        try:
            with shelve.open(self._cache_path) as db:
                entry = db.get(key)
            if entry and time.time() - entry["ts"] < self._cache_ttl:
                return entry["docs"]
        except Exception:
            pass
        return None

    def _cache_set(self, question: str, collection: str, n: int, docs: List["Document"]) -> None:
        if not self._cache_path:
            return
        key = self._cache_key(question, collection, n)
        try:
            with shelve.open(self._cache_path) as db:
                db[key] = {"ts": time.time(), "docs": docs}
        except Exception:
            pass

    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        n: Optional[int] = None,
        collection: Optional[str | list[str]] = None,
    ) -> List[Document]:
        """
        Retrieve documents from SIBILS API.

        When *collection* is a list, each collection is queried independently
        and results are merged (deduplicated by doc_id, sorted by score).

        Args:
            question: User query/question
            n: Number of documents to retrieve (default: self.default_n)
            collection: Override default collection(s).
                        A single string or a list of strings.

        Returns:
            List of Document objects sorted by relevance
        """
        n = n or self.default_n
        collection = collection or self.collection

        # Multi-collection: query all collections in parallel then merge
        if isinstance(collection, list):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            all_docs: Dict[str, Document] = {}
            with ThreadPoolExecutor(max_workers=len(collection)) as executor:
                futures = {executor.submit(self._retrieve_single, question, n, col): col
                           for col in collection}
                for future in as_completed(futures):
                    try:
                        docs = future.result()
                        for doc in docs:
                            # Keep higher-scored duplicate
                            if doc.doc_id not in all_docs or doc.score > all_docs[doc.doc_id].score:
                                all_docs[doc.doc_id] = doc
                    except Exception:
                        continue
            return sorted(all_docs.values(), key=lambda d: d.score, reverse=True)

        return self._retrieve_single(question, n, collection)

    def _retrieve_single(
        self,
        question: str,
        n: int,
        collection: str,
    ) -> List[Document]:
        """Retrieve documents from a single SIBILS collection (cache-aware)."""

        cached = self._cache_get(question, collection, n)
        if cached is not None:
            return cached

        docs = self._retrieve_single_uncached(question, n, collection)
        self._cache_set(question, collection, n, docs)
        return docs

    def _parse_hits(self, data: dict, collection: str) -> List[Document]:
        """Extract Document objects from a SIBILS JSON response."""
        hits = data.get("elastic_output", {}).get("hits", {}).get("hits", [])
        documents = []
        for i, hit in enumerate(hits):
            source = hit.get("_source", {})
            plazi_text = source.get("text") or ""
            plazi_title = source.get("treatment_title") or source.get("article-title") or ""
            documents.append(Document(
                doc_id=hit.get("_id", f"doc_{i}"),
                title=source.get("title") or plazi_title,
                abstract=source.get("abstract") or plazi_text,
                full_text=source.get("full_text"),
                score=float(hit.get("_score") or 0.0),
                source=collection,
                pmid=source.get("pmid"),
                pmcid=source.get("pmcid"),
                doi=source.get("doi"),
            ))
        return documents

    def _retrieve_single_uncached(
        self,
        question: str,
        n: int,
        collection: str,
    ) -> List[Document]:
        """Raw SIBILS API call — no caching."""

        # Parse query if enabled
        parsed_query = None
        use_es_mode = False

        if self.use_query_parser and self.query_parser:
            # Parse with ES query generation if enabled
            parsed_query = self.query_parser.parse(
                question,
                collection=collection,
                include_es_query=self.use_es_query
            )

            # Try ES query first if available
            if self.use_es_query and parsed_query.success and parsed_query.es_query:
                use_es_mode = True

        try:
            docs = None

            if use_es_mode and parsed_query.es_query:
                # Try POST with Elasticsearch query (jq parameter)
                import json
                response = requests.post(
                    self.api_url,
                    params={"col": collection, "n": n},
                    data={"jq": json.dumps(parsed_query.es_query)},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("success", False):
                    docs = self._parse_hits(data, collection)
                # If ES query succeeded but returned 0 hits, fall through to keyword GET below

            if docs is None:
                # Keyword GET: use parsed keywords if available, else raw question
                if parsed_query and parsed_query.success and parsed_query.text_parts:
                    keywords = [p for p in parsed_query.text_parts if p not in ['?', '!', '.', ',']]
                    query_text = ' '.join(keywords) if keywords else question
                else:
                    query_text = question

                response = requests.get(
                    self.api_url,
                    params={"q": query_text, "col": collection, "n": n},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("success", False):
                    error = data.get("error", "Unknown error")
                    raise Exception(f"SIBILS API error: {error}")

                docs = self._parse_hits(data, collection)

            return docs

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
    retriever = SIBILSRetriever()  # defaults to ["medline", "plazi"]

    # Example question
    question = "What is the host of Plasmodium falciparum?"
    print(f"Question: {question}\n")

    # Retrieve documents (searches medline + plazi by default)
    documents = retriever.retrieve(question, n=5)

    print(f"Retrieved {len(documents)} documents:\n")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. [{doc.doc_id}] (score: {doc.score:.2f}, source: {doc.source})")
        print(f"   Title: {doc.title}")
        print(f"   Abstract: {doc.abstract[:200]}...")
        print()


if __name__ == "__main__":
    main()
