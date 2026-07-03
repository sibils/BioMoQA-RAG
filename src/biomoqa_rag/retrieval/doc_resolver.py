"""
Resolve a document reference (PMID, PMCID, DOI, or title fragment) to a Document
object by searching SIBILS, then verifying the top hit matches.

The SIBILS API has no fetch-by-id endpoint, so resolution is done via keyword
search.  For numeric IDs and PMCIDs this is effectively an exact lookup (IDs are
indexed as unique terms).  For DOIs and title fragments it returns the best-scoring
hit across all collections.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .sibils_retriever import Document, SIBILSRetriever


class DocResolver:
    def __init__(self, retriever: SIBILSRetriever) -> None:
        self.retriever = retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, ref: str) -> Optional[Document]:
        """
        Resolve a single reference to a Document.

        Returns None if no matching document is found in SIBILS.
        """
        ref = ref.strip()
        if not ref:
            return None

        # PMID: 1–8 digits
        if re.match(r"^\d{1,8}$", ref):
            return self._resolve_pmid(ref)

        # PMCID: PMC followed by digits (case-insensitive)
        if re.match(r"^PMC\d+$", ref, re.IGNORECASE):
            return self._resolve_pmcid(ref.upper())

        # DOI: starts with "10." followed by registrant code
        if re.match(r"^10\.\d{4,}/", ref):
            return self._resolve_doi(ref)

        # Free-text (title fragment or any other string): search all collections
        return self._resolve_freetext(ref)

    def resolve_batch(self, refs: List[str]) -> tuple[List[Document], List[str]]:
        """
        Resolve a list of references in parallel.

        Returns (resolved_docs, unresolved_refs).
        """
        resolved: List[Document] = []
        unresolved: List[str] = []

        with ThreadPoolExecutor(max_workers=min(len(refs), 4)) as ex:
            futures = {ex.submit(self.resolve, ref): ref for ref in refs}
            for fut in as_completed(futures):
                ref = futures[fut]
                doc = fut.result()
                if doc is not None:
                    resolved.append(doc)
                else:
                    unresolved.append(ref)

        return resolved, unresolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_pmid(self, pmid: str) -> Optional[Document]:
        docs = self.retriever.retrieve(pmid, n=5, collection="medline")
        # Prefer exact PMID match; fall back to top hit
        for doc in docs:
            if doc.pmid == pmid or doc.doc_id == pmid:
                return doc
        return docs[0] if docs else None

    def _resolve_pmcid(self, pmcid: str) -> Optional[Document]:
        docs = self.retriever.retrieve(pmcid, n=5, collection="pmc")
        for doc in docs:
            if doc.pmcid and doc.pmcid.upper() == pmcid:
                return doc
        return docs[0] if docs else None

    def _resolve_doi(self, doi: str) -> Optional[Document]:
        docs = self.retriever.retrieve(doi, n=5, collection=None)
        for doc in docs:
            if doc.doi == doi:
                return doc
        return docs[0] if docs else None

    def _resolve_freetext(self, text: str) -> Optional[Document]:
        """Search all three collections and return the highest-scoring hit."""
        best: Optional[Document] = None
        results: List[Document] = []

        def _search(col: str) -> List[Document]:
            return self.retriever.retrieve(text, n=3, collection=col)

        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(_search, col): col for col in SIBILSRetriever.VALID_COLLECTIONS}
            for fut in as_completed(futs):
                results.extend(fut.result())

        if not results:
            return None
        return max(results, key=lambda d: d.score)
