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
from ..passage import mentions


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
        empty_cache_ttl: int = 0,  # never serve empty (0-result) responses from cache
        enforce_taxon_phrase: bool = True,
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
            empty_cache_ttl: TTL for reading legacy empty (0-result) entries left
                             by older builds. Defaults to 0 so they are never
                             served (they expire on the next read). New empty
                             results are not written to the cache at all.
            enforce_taxon_phrase: When the query parser recognises a multi-word
                             organism name (a binomial such as "Anurida polaris"),
                             require that name as an adjacent phrase in retrieval so
                             the specific epithet alone cannot match a different
                             organism (e.g. the stonefly "Arcynopteryx polaris").
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
        self.enforce_taxon_phrase = enforce_taxon_phrase

        # Initialize query parser if enabled (use first collection for ES query generation)
        if self.use_query_parser:
            first_col = self.collection[0] if isinstance(self.collection, list) else self.collection
            self.query_parser = SIBILSQueryParser(collection=first_col)
        else:
            self.query_parser = None

        # Disk cache
        self._cache_path: Optional[str] = None
        self._cache_ttl: int = cache_ttl
        self._empty_cache_ttl: int = empty_cache_ttl
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
            if entry:
                docs = entry["docs"]
                # Empty results are never served from cache (empty_cache_ttl
                # defaults to 0), so a legacy poisoned empty entry from an older
                # build expires on the next read; real results keep the full TTL.
                ttl = self._cache_ttl if docs else self._empty_cache_ttl
                if time.time() - entry["ts"] < ttl:
                    return docs
        except Exception:
            pass
        return None

    def _cache_set(self, question: str, collection: str, n: int, docs: List["Document"]) -> None:
        if not self._cache_path:
            return
        if not docs:
            # Never cache an empty result: a transient upstream zero (parser blip,
            # ES hiccup) would otherwise be replayed to every subsequent request
            # for the same query. Real results are still cached for cache_ttl.
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

    @staticmethod
    def _with_abstract_filter(es_query: dict) -> dict:
        """
        Return a copy of *es_query* that also requires a non-empty abstract.

        Many Medline citations are title-only (empty abstract). The extractive
        model then 'extracts' the title, producing junk answers like "Disease".
        A `wildcard: {abstract: "*"}` filter drops those at the source while
        preserving recall (an empty-string abstract indexes no terms, so the
        wildcard excludes it). Applied to Medline only — Plazi/PMC carry their
        text in other fields.
        """
        import copy
        q = copy.deepcopy(es_query)
        query = q.get("query")
        abstract_filter = {"wildcard": {"abstract": "*"}}
        if isinstance(query, dict) and "bool" in query:
            query["bool"].setdefault("filter", []).append(abstract_filter)
        else:
            # Wrap whatever query is present in a bool so we can attach the filter
            q["query"] = {
                "bool": {
                    "must": [query] if query else [],
                    "filter": [abstract_filter],
                }
            }
        return q

    # Fields in which a taxon name is required to appear as a phrase, per collection.
    _TAXON_FIELDS = {
        "medline": ["title", "abstract"],
        "pmc":     ["title", "abstract"],
        "plazi":   ["treatment_title", "text", "title"],
    }

    @staticmethod
    def _extract_taxa(es_query: dict) -> list[tuple[str, str]]:
        """
        Find multi-word organism names the query parser recognised.

        The parser tags taxa with an `annotations.all` term of the form
        `ott|<id>|<Name>`. A name containing a space is a binomial (genus +
        epithet); returning it lets retrieval require the whole name rather than
        matching the epithet alone. Returns a list of (name, ott_term) pairs.
        """
        taxa: list[tuple[str, str]] = []

        def walk(obj):
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key == "term" and isinstance(val, dict):
                        ott = val.get("annotations.all", "")
                        if isinstance(ott, str) and ott.startswith("ott|"):
                            parts = ott.split("|", 2)
                            if len(parts) == 3 and " " in parts[2].strip():
                                taxa.append((parts[2].strip(), ott))
                    else:
                        walk(val)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(es_query)
        # dedup preserving order
        seen, out = set(), []
        for name, ott in taxa:
            if name not in seen:
                seen.add(name)
                out.append((name, ott))
        return out

    def _with_taxon_phrase(self, es_query: dict, taxa: list[tuple[str, str]], collection: str) -> dict:
        """
        Return a copy of *es_query* that also requires each recognised binomial to
        appear as an adjacent phrase (or carry the taxon annotation).

        This is added as a filter, so it constrains WHICH documents match without
        changing the relevance scoring produced by the parser. A document qualifies
        if it contains the binomial as a phrase in the collection's text fields OR
        is annotated with that exact taxon — high precision, without losing
        annotation-based recall.
        """
        import copy
        fields = self._TAXON_FIELDS.get(collection, ["title", "abstract"])
        q = copy.deepcopy(es_query)
        query = q.get("query")
        if not (isinstance(query, dict) and "bool" in query):
            query = {"bool": {"must": [query] if query else []}}
            q["query"] = query
        bool_q = query["bool"]
        for name, ott in taxa:
            should = [{"multi_match": {"query": name, "type": "phrase", "fields": fields}}]
            if ott:
                should.append({"term": {"annotations.all": ott}})
            bool_q.setdefault("filter", []).append(
                {"bool": {"should": should, "minimum_should_match": 1}})
        return q

    def _parse_hits(self, data: dict, collection: str) -> List[Document]:
        """Extract Document objects from a SIBILS JSON response."""
        hits = data.get("elastic_output", {}).get("hits", {}).get("hits", [])
        documents = []
        for i, hit in enumerate(hits):
            source = hit.get("_source", {})
            plazi_text = source.get("text") or ""
            plazi_title = source.get("treatment_title") or source.get("article-title") or ""
            doc = Document(
                doc_id=hit.get("_id", f"doc_{i}"),
                title=source.get("title") or plazi_title,
                abstract=source.get("abstract") or plazi_text,
                full_text=source.get("full_text"),
                score=float(hit.get("_score") or 0.0),
                source=collection,
                pmid=source.get("pmid"),
                pmcid=source.get("pmcid"),
                doi=source.get("doi"),
            )
            # Safety net for the keyword-GET fallback (which has no ES filter):
            # drop title-only Medline citations that carry no usable text.
            if collection == "medline" and not (doc.abstract or "").strip() \
                    and not (doc.full_text or "").strip():
                continue
            documents.append(doc)
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

        # Binomials the parser recognised (OTT-annotated), required in every path.
        taxa = []
        if self.enforce_taxon_phrase and parsed_query and parsed_query.success and parsed_query.es_query:
            taxa = self._extract_taxa(parsed_query.es_query)

        try:
            docs = None

            if use_es_mode and parsed_query.es_query:
                # Try POST with Elasticsearch query (jq parameter)
                import json
                es_query = parsed_query.es_query
                if collection == "medline":
                    # Exclude title-only citations (empty abstract) at the source.
                    es_query = self._with_abstract_filter(es_query)
                if taxa:
                    # Require a recognised binomial as a phrase so the specific
                    # epithet alone cannot match a different organism.
                    es_query = self._with_taxon_phrase(es_query, taxa, collection)
                response = requests.post(
                    self.api_url,
                    params={"col": collection, "n": n},
                    data={"jq": json.dumps(es_query)},
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
                if taxa:
                    # The keyword GET has no ES filter, so the binomial is enforced here:
                    # without it the fallback matched the epithet alone, and every wrong
                    # end-to-end body size in D10.3 came from another taxon's document (P3).
                    docs = [d for d in docs
                            if any(mentions(f"{d.title or ''} {d.abstract or ''} {d.full_text or ''}", name)
                                   for name, _ in taxa)]

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
