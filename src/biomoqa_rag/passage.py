"""
Passage selection shared by the generative and the extractive reader.

Both readers see only part of a document: the generative reader `llm_abstract_length`
characters, BioBERT about 512 tokens. Which part they see decides the answer more than
the reader does. In the KYBELE D10.3 evaluation the extractive reader found a body size
96 % of the time when it sat in the first 1,500 characters and 33 % of the time after,
because it was reading the start of the document, not the relevant section.
"""

from __future__ import annotations

import re
from typing import Optional

_STOP = {
    'what', 'is', 'the', 'are', 'of', 'in', 'a', 'an', 'and', 'or',
    'for', 'to', 'does', 'how', 'why', 'which', 'do', 'its', 'by',
}

# Words that open a question and are capitalised for that reason alone.
_NOT_A_GENUS = {
    'what', 'which', 'how', 'where', 'when', 'who', 'whom', 'whose', 'why', 'does', 'do',
    'did', 'is', 'are', 'was', 'were', 'can', 'could', 'should', 'would', 'will', 'the',
    'list', 'name', 'describe', 'give', 'tell', 'find', 'show', 'explain', 'compare',
    'in', 'on', 'for', 'from', 'with', 'about', 'between', 'among', 'during', 'after',
}
# English words that follow a capitalised group name in questions ("Collembola live").
_NOT_AN_EPITHET = {
    'live', 'lives', 'living', 'feed', 'feeds', 'feeding', 'eat', 'eats', 'found', 'occur',
    'occurs', 'inhabit', 'inhabits', 'have', 'has', 'are', 'were', 'was', 'use', 'uses',
    'can', 'from', 'with', 'that', 'this', 'and', 'species', 'genus', 'family', 'populations',
    'diet', 'habitat', 'size', 'body', 'length', 'prefer', 'prefers', 'need', 'needs',
}
_BINOMIAL = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,}(?:-[a-z]+)?)\b")

# Words a treatment section about the trait uses although the question does not. They
# steer the window to the description, measurement or ecology paragraph of a treatment
# instead of whatever paragraph repeats the question's words. They score at half weight
# so that they rank windows but never outweigh the question's own terms.
_TRAIT_CUES = [
    (re.compile(r"\b(?:body (?:length|size)|how (?:long|big|large)|size of|length of)\b", re.I),
     ["length", "size", " mm", "µm", "μm", "body"]),   # " mm": "mm" alone is in "community"
    (re.compile(r"\b(?:habitats?|where (?:does|do|is|are)\b.*\b(?:live|found|occur)|ecology)\b", re.I),
     ["habitat", "ecology", "found", "collected", "litter", "soil", "under ", "moss"]),
    (re.compile(r"\b(?:feed\w*|diet\w*|food|eat\w*|prey|trophic)\b", re.I),
     ["feed", "feeding", "diet", "food", "gut", "fungi", "prey"]),
]


def binomials(question: str) -> list[str]:
    """Genus + epithet pairs named in the question ("Folsomia candida")."""
    out = []
    for g, e in _BINOMIAL.findall(question or ""):
        if g.lower() in _NOT_A_GENUS or e in _STOP or e in _NOT_AN_EPITHET:
            continue
        out.append(f"{g} {e}")
    return list(dict.fromkeys(out))


def taxon_pattern(name: str) -> re.Pattern:
    """The binomial, or its abbreviated form ("F. candida"), as a phrase."""
    genus, epithet = name.split(" ", 1)
    return re.compile(
        rf"\b(?:{re.escape(genus)}|{re.escape(genus[0])}\.)\s*{re.escape(epithet)}\b", re.I)


def mentions(text: str, name: str) -> bool:
    return bool(taxon_pattern(name).search(text or ""))


def trait_cues(question: str) -> set[str]:
    cues: set[str] = set()
    for pat, words in _TRAIT_CUES:
        if pat.search(question or ""):
            cues.update(words)
    return cues


def best_passage(text: str, question: str, window: int,
                 taxa: Optional[list[str]] = None) -> str:
    """Return the `window`-char slice of text with the highest IDF-weighted keyword score.

    For short texts (abstracts) this is a no-op. For full-text documents the relevant
    measurement or description section is often thousands of characters in, so the window
    slides to find it rather than taking the first N characters.

    Uses IDF-style weighting (1/freq) so rare terms like "body" outweigh ubiquitous terms
    like a genus name that appears dozens of times throughout a revision.

    When the question names a species and the text mentions it, only windows that mention
    it are eligible: in a genus revision every other window is about another species.
    Otherwise the rarest question term present anchors the window, as before.
    """
    if len(text) <= window:
        return text
    words = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2 and w not in _STOP}
    if not words:
        return text[:window]
    tl = text.lower()
    # IDF weight: rarer in the document -> higher weight -> more discriminative
    weights = {w: 1.0 / (tl.count(w) + 1) for w in words}
    for c in trait_cues(question) - words:
        weights[c] = 0.5 / (tl.count(c) + 1)

    # Spans that make a window on-topic. A named species beats the rarest term: the
    # epithet alone also matches congeners' abbreviations and other genera.
    spans: list[tuple[int, int]] = []
    for name in (taxa if taxa is not None else binomials(question)):
        spans += [m.span() for m in taxon_pattern(name).finditer(text)]
    anchors: list[str] = []
    if not spans:
        # The rarest term present is what makes a window on-topic: a species epithet
        # inside a genus revision, say. Without this, a comparison table's "Body length"
        # header outranks the single row naming the species actually being asked about.
        present = {w: tl.count(w) for w in words if tl.count(w) > 0}
        if present:
            rarest = min(present.values())
            if rarest <= 3:
                anchors = [w for w, c in present.items() if c == rarest]

    step = max(1, window // 8)
    positions = list(range(0, len(text) - window + 1, step))
    # Windows starting just before each mention, so a mention is never missed by the
    # step and its following description fits in the window.
    positions += [max(0, min(s - window // 10, len(text) - window)) for s, _ in spans]
    best_score, best_pos = -1.0, 0
    best_anchored_score, best_anchored_pos = -1.0, None
    for pos in sorted(set(positions)):
        chunk = tl[pos:pos + window]
        score = 0.0
        for w, wt in weights.items():
            idx = chunk.find(w)
            if idx >= 0:
                # Prefer windows where high-weight keywords appear early: a keyword at
                # the window start scores full weight, at the end half.
                score += wt * (1.0 - 0.5 * idx / window)
        if score > best_score:
            best_score, best_pos = score, pos
        if spans:
            anchored = any(s >= pos and e <= pos + window for s, e in spans)
        else:
            anchored = bool(anchors) and any(a in chunk for a in anchors)
        if anchored and score > best_anchored_score:
            best_anchored_score, best_anchored_pos = score, pos

    # An anchored window always wins over a higher-scoring window that omits it.
    if best_anchored_pos is not None:
        best_pos = best_anchored_pos
    return text[best_pos:best_pos + window]
