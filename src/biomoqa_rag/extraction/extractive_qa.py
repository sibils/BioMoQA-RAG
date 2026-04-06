"""
Extractive QA module using BioBERT fine-tuned on SQuAD2.

Extracts verbatim answer spans from retrieved documents — no generation,
no hallucination. SQuAD2 training enables "no answer" detection.
"""

from typing import List, Dict


class BioExtractiveQA:
    """
    Extractive QA using a BERT-style model trained on SQuAD2.

    Given a question and a list of documents, runs QA inference on all
    documents in a single batched GPU call and returns ALL valid answer spans ranked by score.
    SQuAD2 "impossible answer" outputs a high score with an empty string —
    those are filtered out before ranking so empty-answer docs never win.
    """

    def __init__(
        self,
        model_name: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        confidence_threshold: float = 0.01,
        device: int = -1,
    ):
        """
        Args:
            model_name: HuggingFace model ID (must be SQuAD2-fine-tuned for no-answer support)
            confidence_threshold: Minimum score to return a candidate (0-1)
            device: -1 for CPU, 0+ for GPU index
        """
        from transformers import pipeline

        self.qa = pipeline(
            "question-answering",
            model=model_name,
            handle_impossible_answer=True,
            device=device,
            model_kwargs={"low_cpu_mem_usage": False},
        )
        self.threshold = confidence_threshold

    def extract(self, question: str, documents: List, max_context_length: int = 800) -> List[Dict]:
        """
        Run extractive QA on all documents in a single batched GPU call.

        Returns a list of answer candidates sorted by score descending.
        The list is empty when no document yields a confident answer.

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
        inputs = [{"question": question, "context": ctx} for ctx in contexts]
        results = self.qa(inputs, batch_size=len(inputs))

        # SQuAD2 outputs a HIGH score with an EMPTY string when it decides
        # there is no answer — filter those out before ranking.
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
