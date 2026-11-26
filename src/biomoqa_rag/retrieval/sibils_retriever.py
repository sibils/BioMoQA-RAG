"""
SIBILS API Retriever for Biomedical Documents

This module implements the retrieval stage (R) of the Ragnarok RAG pipeline.
It queries the SIBILS biodiversity/biomedical API to retrieve relevant documents.
"""

import copy
import hashlib
import json
import re
import shelve
import threading
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

    # Part of every cache key. Bump it when what a search returns changes, so results
    # cached by an older build are not replayed for the next cache_ttl (7 days).
    CACHE_VERSION = "2026-11-26"

    # Names the query parser tags as taxa although they are not one.
    JUNK_TAXA = {"Scientific name"}

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
        # One request searches several lists at once; the shelve file takes one at a time.
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, question: str, collection: str, n: int, kind: str = "today") -> str:
        raw = f"{self.CACHE_VERSION}|{kind}|{question.lower().strip()}|{collection}|{n}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, question: str, collection: str, n: int, kind: str = "today") -> Optional[List["Document"]]:
        if not self._cache_path:
            return None
        key = self._cache_key(question, collection, n, kind)
        try:
            with self._cache_lock, shelve.open(self._cache_path) as db:
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

    def _cache_set(self, question: str, collection: str, n: int, docs: List["Document"], kind: str = "today") -> None:
        if not self._cache_path:
            return
        if not docs:
            # Never cache an empty result: a transient upstream zero (parser blip,
            # ES hiccup) would otherwise be replayed to every subsequent request
            # for the same query. Real results are still cached for cache_ttl.
            return
        key = self._cache_key(question, collection, n, kind)
        try:
            with self._cache_lock, shelve.open(self._cache_path) as db:
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
                            name = parts[2].strip() if len(parts) == 3 else ""
                            if " " in name and name not in SIBILSRetriever.JUNK_TAXA:
                                taxa.append((name, ott))
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

    def _taxon_filter(self, taxa: list[tuple[str, str]], collection: str) -> dict:
        """The species filter: a document must name one of the recognised binomials as a
        phrase in the collection's text fields, or carry that exact taxon annotation.

        One of them, not all: a question naming four species was required to find all four
        in one abstract and found nothing. The epithet alone still never qualifies.
        """
        fields = self._TAXON_FIELDS.get(collection, ["title", "abstract"])
        should = []
        for name, ott in taxa:
            should.append({"multi_match": {"query": name, "type": "phrase", "fields": fields}})
            if ott:
                should.append({"term": {"annotations.all": ott}})
        return {"bool": {"should": should, "minimum_should_match": 1}}

    def _with_taxon_phrase(self, es_query: dict, taxa: list[tuple[str, str]], collection: str) -> dict:
        """
        Return a copy of *es_query* that also requires one of the recognised binomials
        as an adjacent phrase (or the taxon annotation).

        This is added as a filter, so it constrains WHICH documents match without
        changing the relevance scoring produced by the parser.
        """
        q = copy.deepcopy(es_query)
        query = q.get("query")
        if not (isinstance(query, dict) and "bool" in query):
            query = {"bool": {"must": [query] if query else []}}
            q["query"] = query
        query["bool"].setdefault("filter", []).append(self._taxon_filter(taxa, collection))
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

    # Brackets and quotes glued to a word make the query parser reject the question
    # (HTTP 400): 16 of 258 test questions, such as "adriamycin(doxorubicin)?".
    _PARSER_BREAKERS = re.compile(r'[()\[\]"]')

    def parse(self, question: str, collection: str):
        """The parser's reading of the question, and the binomials it recognised.

        A question the parser rejects is tried once more with its brackets and quotes
        replaced by spaces; that gave documents for 15 of the 16 rejected test questions.
        Returns (parsed_query or None, taxa, the text that was parsed).
        """
        if not (self.use_query_parser and self.query_parser):
            return None, [], question
        text = question
        parsed = self.query_parser.parse(text, collection=collection, include_es_query=self.use_es_query)
        if not parsed.success and self._PARSER_BREAKERS.search(question):
            text = " ".join(self._PARSER_BREAKERS.sub(" ", question).split())
            parsed = self.query_parser.parse(text, collection=collection, include_es_query=self.use_es_query)
        taxa = []
        if self.enforce_taxon_phrase and parsed.success and parsed.es_query:
            taxa = self._extract_taxa(parsed.es_query)
        return parsed, taxa, text

    def _post(self, es_query: dict, collection: str, n: int, timeout: float) -> List[Document]:
        response = requests.post(
            self.api_url,
            params={"col": collection, "n": n},
            data={"jq": json.dumps(es_query)},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success", False):
            raise Exception(f"SIBILS API error: {data.get('error', 'Unknown error')}")
        return self._parse_hits(data, collection)

    def _retrieve_single_uncached(
        self,
        question: str,
        n: int,
        collection: str,
        parsed=None,
    ) -> List[Document]:
        """Today's query: the parser's query for the collection, with the species filter.

        Falls back to a keyword GET when the parser gives no query or the query finds
        nothing. Raw SIBILS call, no caching. `parsed` = a result of parse(), to reuse.
        """
        parsed_query, taxa, text = parsed if parsed is not None else self.parse(question, collection)
        use_es_mode = bool(self.use_es_query and parsed_query is not None
                           and parsed_query.success and parsed_query.es_query)

        try:
            docs = None

            if use_es_mode:
                # Medline used to add a `wildcard abstract:*` filter here to drop title-only
                # citations. It cost a median 1.8 s per lookup, and _parse_hits drops those
                # citations anyway: the delivered lists were identical for 242 of 242 questions.
                es_query = parsed_query.es_query
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
                # Keyword GET: use parsed keywords if available, else the (bracket-free) text
                if parsed_query is not None and parsed_query.success and parsed_query.text_parts:
                    keywords = [p for p in parsed_query.text_parts if p not in ['?', '!', '.', ',']]
                    query_text = ' '.join(keywords) if keywords else text
                else:
                    query_text = text

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

    # ------------------------------------------------------------------
    # The lists the depth pipeline merges
    # ------------------------------------------------------------------

    # Fields a plain OR search matches, per collection.
    _PLAIN_FIELDS = {
        "medline": ["title", "abstract"],
        "pmc":     ["title", "abstract"],
        "plazi":   ["treatment_title", "title", "text"],
    }

    def today_list(self, question: str, collection: str, n: int, parsed=None) -> List[Document]:
        """Today's query (cache-aware). `parsed` = a result of parse(), to reuse."""
        cached = self._cache_get(question, collection, n, "today")
        if cached is not None:
            return cached
        docs = self._retrieve_single_uncached(question, n, collection, parsed=parsed)
        self._cache_set(question, collection, n, docs, "today")
        return docs

    def plain_search(
        self,
        text: str,
        collection: str,
        n: int,
        taxa: list[tuple[str, str]],
        timeout: Optional[float] = None,
    ) -> List[Document]:
        """Any of the words of `text`, ranked by BM25, with the species filter (cache-aware).

        With a `timeout`, a search that fails or takes longer returns an empty list:
        that is how the optional searches of the deep mode are run.
        """
        words = " ".join(re.sub(r"[^\w\s\-\.]", " ", text or "").split())
        if not words:
            return []
        names = "|".join(sorted(name for name, _ in taxa))
        kind = f"or|{names}"
        cached = self._cache_get(words, collection, n, kind)
        if cached is not None:
            return cached
        fields = self._PLAIN_FIELDS.get(collection, ["title", "abstract"])
        query: dict = {"multi_match": {"query": words, "fields": fields, "operator": "or"}}
        if taxa:
            query = {"bool": {"must": [query], "filter": [self._taxon_filter(taxa, collection)]}}
        try:
            docs = self._post({"query": query}, collection, n, timeout or self.timeout)
        except Exception:
            if timeout is not None:
                return []
            raise
        self._cache_set(words, collection, n, docs, kind)
        return docs

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
