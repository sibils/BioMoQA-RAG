"""
What an answer says about itself, and how collections are ordered by their answers.

Generative answers carry no span score, but their wording says a lot: in the KYBELE D10.3
evaluation hedged answers were right 76-82 % of the time against 91-98 % for plain ones,
and an answer saying the sources do not state the trait was listed first whenever its
collection sorted first, displacing a treatment that did state it.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

from .passage import binomials

# The answer says the sources do not contain what was asked.
_NO_INFORMATION = re.compile(
    r"\b(?:no|not any|little) (?:specific |explicit |direct |detailed |relevant )?"
    r"(?:information|data|details?|mention|description|evidence)\b"
    r"|\b(?:is|are|was|were) not (?:explicitly |specifically |directly )?"
    r"(?:mentioned|stated|provided|specified|described|reported|given|available|discussed)\b"
    r"|\b(?:do|does|did) not (?:explicitly |specifically |directly )?"
    r"(?:mention|state|provide|specify|describe|report|contain|include|discuss|address)\b"
    r"|\b(?:lack|lacks) (?:sufficient |specific )?(?:information|details?)\b"
    r"|\binsufficient (?:information|data)\b"
    r"|\bcannot be determined\b",
    re.I)

# The answer commits only partly.
_HEDGE = re.compile(
    r"\b(?:may|might|could|possibly|probably|likely|presumably|perhaps|apparently"
    r"|suggests?|suggested|appears? to|seems? to|is thought to|it is unclear|uncertain)\b",
    re.I)

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*(])")


def states_no_answer(answer: str) -> bool:
    """True when the whole answer amounts to "the sources do not say".

    An answer that states a value and closes with "other food sources are not mentioned"
    has an answer: only the first sentence decides, where the model puts its verdict.
    """
    a = (answer or "").strip()
    if not a:
        return True
    first = _SENT.split(a, maxsplit=1)[0]
    return bool(_NO_INFORMATION.search(first))


def is_hedged(answer: str) -> bool:
    return bool(_HEDGE.search(answer or ""))


def mean_token_probability(logprobs: Optional[Iterable[float]]) -> Optional[float]:
    """Geometric mean of the generated tokens' probabilities, in [0, 1]."""
    vals = [lp for lp in (logprobs or []) if lp is not None and not math.isinf(lp)]
    if not vals:
        return None
    return round(math.exp(sum(vals) / len(vals)), 4)


def annotate(answer_item: dict) -> dict:
    """Add `hedged` and `states_no_answer` to an answer dict, in place."""
    text = answer_item.get("answer") or ""
    answer_item["states_no_answer"] = states_no_answer(text)
    answer_item["hedged"] = is_hedged(text) and not answer_item["states_no_answer"]
    return answer_item


def rank_collection_results(question: str, results: list[dict], use_scores: bool = True) -> list[dict]:
    """Order collection results best answer first, and renumber `rank`.

    1. a collection with an answer before one without;
    2. an answer that answers before one saying the sources do not state it;
    3. for a question about a named species, Plazi first: a taxonomic treatment is the
       primary description of the species (D10.3: the PMC answer came first for 94 of
       126 trait questions);
    4. a plain answer before a hedged one;
    5. with `use_scores`, the higher span score (extractive) or token probability
       (generative): for doc_refs, where there is no retrieval order to fall back on;
    6. the order the results came in.
    """
    taxon_question = bool(binomials(question))

    def key(item):
        i, r = item
        top = (r.get("answers") or [None])[0]
        if not top or not (top.get("answer") or "").strip():
            return (1, 1, 1, 1, 0.0, i)
        no_answer = top.get("states_no_answer")
        if no_answer is None:
            no_answer = states_no_answer(top["answer"])
        hedged = top.get("hedged")
        if hedged is None:
            hedged = is_hedged(top["answer"])
        score = top.get("answer_score")
        if score is None:
            score = top.get("generation_confidence") or 0.0
        if not use_scores:
            score = 0.0
        plazi_last = int(not (taxon_question and r.get("collection") == "plazi"))
        return (0, int(bool(no_answer)), plazi_last, int(bool(hedged)), -float(score), i)

    ordered = [r for _, r in sorted(enumerate(results), key=key)]
    return [{**r, "rank": n} for n, r in enumerate(ordered, start=1)]
