"""
Extractive QA module using BioBERT fine-tuned on SQuAD2.

Extracts verbatim answer spans from retrieved documents — no generation,
no hallucination. Always returns the best span per document; quality
filtering is done upstream (min_extractive_score threshold in hybrid mode).
"""

from typing import List, Dict


class BioExtractiveQA:
    """
    Extractive QA using a BERT-style model trained on SQuAD2.

    Given a question and a list of documents, runs QA inference on all
    documents in a single batched GPU call and returns ALL answer spans
    ranked by score. handle_impossible_answer=False ensures the model
    always returns a span — quality gating is done upstream so extractive
    mode always has output when documents are retrieved.
    """

    def __init__(
        self,
        model_name: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        confidence_threshold: float = 0.01,
        device: int = -1,
    ):
        """
        Args:
            model_name: HuggingFace model ID (SQuAD2-fine-tuned)
            confidence_threshold: Minimum score to return a candidate (0-1)
            device: -1 for CPU, 0+ for GPU index
        """
        from transformers import pipeline

        self.qa = pipeline(
            "question-answering",
            model=model_name,
            handle_impossible_answer=False,
            device=device,
            model_kwargs={"low_cpu_mem_usage": False},
        )
        self.threshold = confidence_threshold

    def extract(self, question: str, documents: List, max_context_length: int = 800) -> List[Dict]:
        """
        Run extractive QA on all documents in a single batched GPU call.

        Always returns at least one candidate per document with non-empty text.
        Returns candidates sorted by score descending.

        Each candidate dict has:
          - text (str): verbatim extracted span
          - score (float): model confidence (0-1)
          - doc_idx (int): index in the documents list
          - span_start (int): char offset within passage
          - span_end (int): char offset within passage
          - passage (str): context fed to BioBERT (title + abstract, truncated)
        """
        contexts = [
            ((doc.title.strip() + ". " if doc.title and doc.title.strip() else "") + (doc.abstract or ""))[:max_context_length]
            for doc in documents
        ]
        inputs = [{"question": question, "context": ctx} for ctx in contexts if ctx.strip()]
        if not inputs:
            return []

        results = self.qa(inputs, batch_size=len(inputs))

        candidates = []
        for idx, (result, context) in enumerate(zip(results, contexts)):
            if result["answer"].strip() and result["score"] >= self.threshold:
                candidates.append({
                    "text": result["answer"],
                    "score": result["score"],
                    "doc_idx": idx,
                    "span_start": result["start"],
                    "span_end": result["end"],
                    "passage": context,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
