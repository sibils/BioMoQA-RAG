"""
Extractive QA module using BioBERT fine-tuned on SQuAD2.

Single-pass strategy matching the original sibils.org QA behaviour:
one forward pass per document on the first 512 tokens, no sliding window.
Quality filtering is done upstream (min_extractive_score threshold in hybrid mode).
"""

from typing import List, Dict

# Context window: ~2000 chars ≈ 500 tokens, safely within one BioBERT window (512 max).
_CONTEXT_LEN = 2000


class BioExtractiveQA:
    """
    Extractive QA using a BERT-style model trained on SQuAD2.

    Single-pass strategy: one BioBERT forward pass per document on the
    first 2000 chars (~500 tokens). Matches original sibils.org QA behaviour.

    handle_impossible_answer=False ensures a span is always returned.
    Quality gating is done upstream via min_extractive_score in hybrid mode.
    """

    def __init__(
        self,
        model_name: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        confidence_threshold: float = 0.0,
        device: int = -1,
    ):
        from transformers import pipeline

        self.qa = pipeline(
            "question-answering",
            model=model_name,
            handle_impossible_answer=False,
            device=device,
            model_kwargs={"low_cpu_mem_usage": False},
        )

    def extract(self, question: str, documents: List, max_context_length: int = 800) -> List[Dict]:
        """
        Run single-pass extractive QA over documents.

        Returns candidates sorted by score descending. Always returns at least
        one candidate per document when documents have non-empty text.

        Each candidate dict has:
          - text (str): verbatim extracted span
          - score (float): model confidence (0-1)
          - doc_idx (int): index into the documents list
          - span_start / span_end (int): char offsets within passage
          - passage (str): context actually fed to BioBERT
        """
        full_contexts = [
            ((doc.title.strip() + ". " if doc.title and doc.title.strip() else "") + (doc.abstract or ""))
            for doc in documents
        ]
        valid = [(i, ctx[:_CONTEXT_LEN]) for i, ctx in enumerate(full_contexts) if ctx.strip()]
        if not valid:
            return []

        orig_indices, contexts = zip(*valid)

        inputs = [{"question": question, "context": ctx} for ctx in contexts]
        results = self.qa(inputs, batch_size=len(inputs))
        # HF pipeline returns a plain dict (not a list) when given a single input — normalise.
        if isinstance(results, dict):
            results = [results]

        candidates = []
        for orig_idx, ctx, result in zip(orig_indices, contexts, results):
            if result["answer"].strip():
                candidates.append({
                    "text": result["answer"],
                    "score": result["score"],
                    "doc_idx": orig_idx,
                    "span_start": result["start"],
                    "span_end": result["end"],
                    "passage": ctx,
                })

        return sorted(candidates, key=lambda x: x["score"], reverse=True)
