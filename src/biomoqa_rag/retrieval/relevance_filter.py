"""
Relevance Filtering for Retrieved Documents

Filters out irrelevant documents before passing to LLM generation.
This reduces context pollution and improves answer quality.
"""

from typing import List
import re

from . import Document


class LLMRelevanceFilter:
    """
    Use LLM to filter out irrelevant documents.

    Asks the LLM: "Does this document help answer the question?"
    Only keeps documents where LLM says YES.
    """

    def __init__(self, llm=None, min_relevant: int = 5):
        """
        Initialize relevance filter.

        Args:
            llm: vLLM LLM instance
            min_relevant: Minimum number of relevant docs to keep (even if LLM says no)
        """
        self.llm = llm
        self.min_relevant = min_relevant

    def filter_relevant(
        self,
        question: str,
        documents: List[Document],
        max_docs: int = 20
    ) -> List[Document]:
        """
        Filter documents to only relevant ones.

        Args:
            question: The query question
            documents: Retrieved documents
            max_docs: Maximum documents to return

        Returns:
            Filtered list of relevant documents
        """
        if self.llm is None or len(documents) == 0:
            return documents[:max_docs]

        relevant_docs = []
        relevance_scores = []

        from vllm import SamplingParams
        filter_params = SamplingParams(
            temperature=0.0,  # Deterministic
            max_tokens=10,
        )

        # Check each document
        for doc in documents:
            prompt = f"""Question: {question}

Document title: {doc.title}
Document abstract: {doc.abstract[:400]}

Is this document relevant for answering the question?
Answer only: YES or NO

Answer:"""

            # Generate
            outputs = self.llm.generate([prompt], filter_params)
            response = outputs[0].outputs[0].text.strip().upper()

            # Check response
            is_relevant = "YES" in response or "RELEVANT" in response

            if is_relevant:
                relevant_docs.append(doc)

            # Track for fallback
            relevance_scores.append((1 if is_relevant else 0, doc))

            # Stop if we have enough relevant docs
            if len(relevant_docs) >= max_docs:
                break

        # Fallback: if too few relevant, take top docs anyway
        if len(relevant_docs) < self.min_relevant:
            # Sort by relevance score, then original order
            relevance_scores.sort(reverse=True, key=lambda x: x[0])
            relevant_docs = [doc for score, doc in relevance_scores[:max_docs]]

        return relevant_docs[:max_docs]


class FastRelevanceFilter:
    """
    Fast keyword-based relevance filtering.

    Checks if document contains key terms from the question.
    Uses prefix matching to handle plurals and suffixes (e.g.
    "earthworm" matches "earthworms", "correlat" matches "correlated").
    Always returns at least max_docs documents as fallback so BioBERT
    is never left with an empty context.
    """

    STOP_WORDS = {
        'what', 'is', 'the', 'are', 'of', 'in', 'a', 'an', 'and', 'or',
        'for', 'to', 'from', 'by', 'with', 'at', 'on', 'how', 'why',
        'when', 'where', 'who', 'which', 'does', 'do', 'can', 'has',
        'have', 'been', 'was', 'were', 'that', 'this', 'it', 'its',
    }

    def __init__(self, min_overlap: float = 0.15):
        self.min_overlap = min_overlap

    def filter_relevant(
        self,
        question: str,
        documents: List[Document],
        max_docs: int = 20
    ) -> List[Document]:
        if not documents:
            return []

        # Extract keywords — truncate to 5 chars as a simple stemming proxy
        # (e.g. "earthworms" → "earth", "correlated" → "corre")
        # Use full word for short keywords (≤ 5 chars)
        keywords = []
        for w in re.findall(r'\w+', question):
            w = w.lower()
            if w not in self.STOP_WORDS and len(w) > 2:
                keywords.append(w[:6] if len(w) > 6 else w)

        if not keywords:
            return documents[:max_docs]

        # Score each document by fraction of keywords found (prefix match)
        scored = []
        for doc in documents:
            doc_text = f"{doc.title} {doc.abstract}".lower()
            matches = sum(1 for kw in keywords if kw in doc_text)
            score = matches / len(keywords)
            scored.append((score, doc))

        scored.sort(reverse=True, key=lambda x: x[0])

        # Keep docs above threshold; always return at least max_docs as fallback
        above = [doc for score, doc in scored if score >= self.min_overlap]
        if len(above) >= max_docs:
            return above[:max_docs]

        # Fallback: pad with next best docs to reach max_docs
        below = [doc for score, doc in scored if score < self.min_overlap]
        return (above + below)[:max_docs]


class HybridRelevanceFilter:
    """
    Combine fast keyword filtering with LLM verification.

    First pass: Fast keyword filter
    Second pass: LLM verification on remaining docs
    """

    def __init__(self, llm=None, min_overlap: float = 0.15, min_relevant: int = 5):
        """Initialize hybrid filter."""
        self.fast_filter = FastRelevanceFilter(min_overlap=min_overlap)
        self.llm_filter = LLMRelevanceFilter(llm=llm, min_relevant=min_relevant)

    def filter_relevant(
        self,
        question: str,
        documents: List[Document],
        max_docs: int = 20
    ) -> List[Document]:
        """Two-stage filtering."""
        # Stage 1: Fast keyword filter (keep 2x what we need)
        candidates = self.fast_filter.filter_relevant(
            question, documents, max_docs=max_docs * 2
        )

        # Stage 2: LLM verification
        relevant = self.llm_filter.filter_relevant(
            question, candidates, max_docs=max_docs
        )

        return relevant
