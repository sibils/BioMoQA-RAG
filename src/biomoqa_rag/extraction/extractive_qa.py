"""
Extractive QA module using BioBERT fine-tuned on SQuAD2.

A document of up to 8,000 characters (a Plazi treatment, an abstract) is read whole; a
longer one (PMC full text) is cut to its most relevant passage with the same anchored
selection the generative reader uses. The text is read in overlapping 512-token windows.
Before, one forward pass read the first 512 tokens of a 2,000-character window, so
about 1,500 characters were read and the rest was silently truncated: in the KYBELE
D10.3 evaluation a body size was found 96 % of the time before 1,500 characters and
33 % of the time after.
"""

from typing import List, Dict

from ..passage import best_passage

_READ_WHOLE_UPTO = 8000   # documents up to this many characters are read whole
_CONTEXT_LEN = 4000       # otherwise, the anchored passage of this many characters
_MAX_SEQ_LEN = 512        # BioBERT's limit, question included
_DOC_STRIDE = 128         # token overlap between consecutive windows
_MAX_ANSWER_TOKENS = 30


class BioExtractiveQA:
    """
    Extractive QA using a BERT-style model trained on SQuAD2.

    One batched forward pass per document over the overlapping windows of its
    passage; the best span over all windows is the document's answer.
    """

    def __init__(
        self,
        model_name: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        confidence_threshold: float = 0.0,
        device: int = -1,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, low_cpu_mem_usage=False
        )
        self.torch_device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self.model.to(self.torch_device)
        self.model.eval()

    def _run_one(self, question: str, context: str) -> Dict:
        import torch
        import torch.nn.functional as F

        enc = self.tokenizer(
            question,
            context,
            max_length=_MAX_SEQ_LEN,
            truncation="only_second",
            stride=_DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        offsets_all = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)
        n_windows = offsets_all.shape[0]
        # Context tokens only (sequence id 1): the span must not be in the question,
        # in a special token or in padding.
        ctx_masks = [
            torch.tensor([sid == 1 for sid in enc.sequence_ids(w)], dtype=torch.bool)
            for w in range(n_windows)
        ]
        inputs = {k: v.to(self.torch_device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**inputs)

        best = {"answer": "", "score": 0.0, "start": 0, "end": 0}
        for w in range(n_windows):
            mask = ctx_masks[w].to(out.start_logits.device)
            if not bool(mask.any()):
                continue
            # The [CLS] position is SQuAD2's "no answer". Keeping it in the normalisation
            # means a window that does not contain the answer gives its spans little
            # probability, so spans from different windows can be compared.
            norm = mask.clone()
            norm[0] = True
            start_probs = F.softmax(out.start_logits[w].masked_fill(~norm, float("-inf")), dim=-1)
            end_probs = F.softmax(out.end_logits[w].masked_fill(~norm, float("-inf")), dim=-1)
            start_probs = start_probs.masked_fill(~mask, 0.0)
            end_probs = end_probs.masked_fill(~mask, 0.0)

            # Best valid span: end >= start, at most _MAX_ANSWER_TOKENS tokens.
            n = len(start_probs)
            score_matrix = torch.triu(start_probs.unsqueeze(1) * end_probs.unsqueeze(0))
            score_matrix = torch.tril(score_matrix, diagonal=_MAX_ANSWER_TOKENS)
            flat = score_matrix.view(-1).argmax().item()
            s_tok, e_tok = flat // n, flat % n
            score = score_matrix[s_tok, e_tok].item()
            if score <= best["score"]:
                continue
            start_char = offsets_all[w][s_tok][0].item()
            end_char = offsets_all[w][e_tok][1].item()
            answer = context[start_char:end_char].strip()
            if answer:
                best = {"answer": answer, "score": score, "start": start_char, "end": end_char}
        return best

    @staticmethod
    def _best_passage(text: str, question: str, window: int) -> str:
        """The anchored passage selection the generative reader uses (passage.best_passage)."""
        return best_passage(text, question, window)

    def extract(self, question: str, documents: List, max_context_length: int = 800) -> List[Dict]:
        """
        Run single-pass extractive QA over documents.

        Returns candidates sorted by score descending.
        """
        full_contexts = [
            (doc.title.strip() + ". " if doc.title and doc.title.strip() else "")
            + (doc.full_text or doc.abstract or "")
            for doc in documents
        ]
        valid = [(i, ctx if len(ctx) <= _READ_WHOLE_UPTO
                  else self._best_passage(ctx, question, _CONTEXT_LEN))
                 for i, ctx in enumerate(full_contexts) if ctx.strip()]
        if not valid:
            return []

        candidates = []
        for orig_idx, ctx in valid:
            result = self._run_one(question, ctx)
            if result["answer"]:
                candidates.append({
                    "text": result["answer"],
                    "score": result["score"],
                    "doc_idx": orig_idx,
                    "span_start": result["start"],
                    "span_end": result["end"],
                    "passage": ctx,
                })

        return sorted(candidates, key=lambda x: x["score"], reverse=True)
