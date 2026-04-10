"""
Extractive QA module using BioBERT fine-tuned on SQuAD2.

Extracts verbatim answer spans from retrieved documents — no generation,
no hallucination. Always returns the best span per document; quality
filtering is done upstream (min_extractive_score threshold in hybrid mode).
"""

from typing import List, Dict

# Fast pass: cap context to this many chars (~2 BioBERT windows).
# Covers most abstracts fully. Only pay the sliding-window cost when
# the fast pass finds nothing confident.
_FAST_CONTEXT_LEN = 2000
_FAST_SCORE_MIN = 0.01  # if best score >= this on fast pass, skip full scan


class BioExtractiveQA:
    """
    Extractive QA using a BERT-style model trained on SQuAD2.

    Two-pass strategy for speed:
    1. Fast pass: truncate context to 2000 chars (~1-2 BioBERT windows).
       If any candidate scores >= 0.01, return those results.
    2. Full pass: if fast pass finds nothing, run sliding window over
       the full document text (handles long plazi/pmc full-text docs).

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

    def _run(self, question: str, contexts: List[str], max_seq_len: int = 512, doc_stride: int = 128) -> List[dict]:
        inputs = [{"question": question, "context": ctx} for ctx in contexts]
        return self.qa(inputs, batch_size=len(inputs), max_seq_len=max_seq_len, doc_stride=doc_stride)

    def extract(self, question: str, documents: List, max_context_length: int = 800) -> List[Dict]:
        """
        Run extractive QA with two-pass strategy: fast truncated pass first,
        full sliding-window pass only for docs that returned no confident answer.

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
        # Filter empty docs but track original indices
        valid = [(i, ctx) for i, ctx in enumerate(full_contexts) if ctx.strip()]
        if not valid:
            return []

        orig_indices, contexts = zip(*valid)

        # ── Pass 1: fast truncated contexts ──────────────────────────────
        fast_contexts = [ctx[:_FAST_CONTEXT_LEN] for ctx in contexts]
        fast_results = self._run(question, list(fast_contexts))

        candidates = {}  # orig_idx -> best candidate
        needs_full = []  # indices (into valid) that need full pass

        for vi, (result, fast_ctx) in enumerate(zip(fast_results, fast_contexts)):
            orig_idx = orig_indices[vi]
            if result["answer"].strip() and result["score"] >= _FAST_SCORE_MIN:
                candidates[orig_idx] = {
                    "text": result["answer"],
                    "score": result["score"],
                    "doc_idx": orig_idx,
                    "span_start": result["start"],
                    "span_end": result["end"],
                    "passage": fast_ctx,
                }
            else:
                needs_full.append(vi)

        # ── Pass 2: full sliding window for docs with no confident answer ─
        if needs_full:
            full_inputs = [contexts[vi] for vi in needs_full]
            full_results = self._run(question, full_inputs)
            for vi, result in zip(needs_full, full_results):
                orig_idx = orig_indices[vi]
                full_ctx = contexts[vi]
                if result["answer"].strip():
                    candidates[orig_idx] = {
                        "text": result["answer"],
                        "score": result["score"],
                        "doc_idx": orig_idx,
                        "span_start": result["start"],
                        "span_end": result["end"],
                        "passage": full_ctx,
                    }

        result_list = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return result_list
