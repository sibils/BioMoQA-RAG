"""
Extractive QA module using BioBERT fine-tuned on SQuAD2.

Extracts verbatim answer spans from retrieved documents — no generation,
no hallucination. SQuAD2 training enables "no answer" detection.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict


class BioExtractiveQA:
    """
    Extractive QA using a BERT-style model trained on SQuAD2.

    Given a question and a list of documents, runs QA inference on each
    document in parallel and returns the best verbatim span found.
    If the best score is below the confidence threshold, reports no answer.
    """

    def __init__(
        self,
        model_name: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        confidence_threshold: float = 0.1,
        device: int = -1,
    ):
        """
        Args:
            model_name: HuggingFace model ID (must be SQuAD2-fine-tuned for no-answer support)
            confidence_threshold: Minimum score to consider an answer valid (0-1)
            device: -1 for CPU, 0+ for GPU index
        """
        from transformers import pipeline

        self.qa = pipeline(
            "question-answering",
            model=model_name,
            handle_impossible_answer=True,
            device=device,
        )
        self.threshold = confidence_threshold

    def extract(self, question: str, documents: List, max_context_length: int = 800) -> Dict:
        """
        Run extractive QA on all documents in parallel and return the best span.

        Args:
            question: The biomedical question
            documents: Retrieved document objects (must have .title and .abstract)
            max_context_length: Max characters for context (title + abstract)

        Returns:
            Dict with keys:
              - is_answered (bool)
              - text (str|None): verbatim extracted span
              - doc_idx (int|None): index of source document in documents list
              - score (float): model confidence score
        """
        def _query_doc(idx_doc):
            idx, doc = idx_doc
            context = f"{doc.title}. {doc.abstract}"[:max_context_length]
            result = self.qa(question=question, context=context)
            return idx, result  # result keys: 'answer', 'score', 'start', 'end'

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_query_doc, enumerate(documents)))

        # Pick the span with the highest confidence across all docs
        best_idx, best_result = max(results, key=lambda x: x[1]["score"])

        if best_result["score"] < self.threshold or not best_result["answer"].strip():
            return {"is_answered": False, "text": None, "doc_idx": None, "score": 0.0}

        return {
            "is_answered": True,
            "text": best_result["answer"],
            "doc_idx": best_idx,
            "score": best_result["score"],
        }
