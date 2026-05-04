"""
Extractive QA module using BioBERT fine-tuned on SQuAD2.

Single-pass strategy matching the original sibils.org QA behaviour:
one forward pass per document on the first 512 tokens, no sliding window.
"""

from typing import List, Dict

_CONTEXT_LEN = 2000
_MAX_ANSWER_TOKENS = 30


class BioExtractiveQA:
    """
    Extractive QA using a BERT-style model trained on SQuAD2.

    Single-pass strategy: one BioBERT forward pass per document on the
    first 2000 chars (~500 tokens). Matches original sibils.org QA behaviour.
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
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0]
        enc = {k: v.to(self.torch_device) for k, v in enc.items()}
        token_type_ids = enc.get("token_type_ids")
        with torch.no_grad():
            out = self.model(**enc)

        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]

        # Mask question tokens (token_type_id == 0) so the span must be in the context.
        if token_type_ids is not None:
            ctx_mask = token_type_ids[0].bool()
            start_logits = start_logits.masked_fill(~ctx_mask, float("-inf"))
            end_logits = end_logits.masked_fill(~ctx_mask, float("-inf"))

        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        # Best valid span: end >= start, max span _MAX_ANSWER_TOKENS tokens.
        n = len(start_logits)
        score_matrix = start_probs.unsqueeze(1) * end_probs.unsqueeze(0)  # [n, n]
        for i in range(n):
            lo = max(0, i - _MAX_ANSWER_TOKENS)
            score_matrix[i, :lo] = 0.0
            score_matrix[i, :i] = 0.0

        flat = score_matrix.view(-1).argmax().item()
        best_start = flat // n
        best_end = flat % n
        best_score = score_matrix[best_start, best_end].item()

        start_char = offsets[best_start][0].item()
        end_char = offsets[best_end][1].item()
        answer = context[start_char:end_char].strip()

        return {"answer": answer, "score": best_score, "start": start_char, "end": end_char}

    def extract(self, question: str, documents: List, max_context_length: int = 800) -> List[Dict]:
        """
        Run single-pass extractive QA over documents.

        Returns candidates sorted by score descending.
        """
        full_contexts = [
            ((doc.title.strip() + ". " if doc.title and doc.title.strip() else "") + (doc.abstract or ""))
            for doc in documents
        ]
        valid = [(i, ctx[:_CONTEXT_LEN]) for i, ctx in enumerate(full_contexts) if ctx.strip()]
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
