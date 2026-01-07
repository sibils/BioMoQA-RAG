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
    Much faster than LLM-based filtering but less accurate.
    """

    def __init__(self, min_overlap: float = 0.2):
        """
        Initialize fast filter.

        Args:
            min_overlap: Minimum fraction of question keywords in document
        """
        self.min_overlap = min_overlap

    def filter_relevant(
        self,
        question: str,
        documents: List[Document],
        max_docs: int = 20
    ) -> List[Document]:
        """Filter by keyword overlap."""
        if len(documents) == 0:
            return []

        # Extract keywords from question (remove stop words)
        stop_words = {'what', 'is', 'the', 'are', 'of', 'in', 'a', 'an', 'and',
                     'or', 'for', 'to', 'from', 'by', 'with', 'at', 'on'}

        question_words = set(
            w.lower() for w in re.findall(r'\w+', question)
            if w.lower() not in stop_words and len(w) > 2
        )

        if not question_words:
            return documents[:max_docs]

        # Score documents by keyword overlap
        scored_docs = []
        for doc in documents:
            doc_text = f"{doc.title} {doc.abstract}".lower()
            doc_words = set(re.findall(r'\w+', doc_text))

            overlap = len(question_words & doc_words) / len(question_words)

            if overlap >= self.min_overlap:
                scored_docs.append((overlap, doc))

        # Sort by overlap (descending)
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Return top max_docs
        return [doc for score, doc in scored_docs[:max_docs]]


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
