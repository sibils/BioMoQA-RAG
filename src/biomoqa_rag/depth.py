"""
Search depth: how hard the service looks for the five documents the reader gets.

  fast  Two keyword searches per collection, run side by side: the query parser's query
        (today's list) and a plain OR search over the question's words. Both keep the
        species filter. A cross-encoder scores every candidate against the question as
        asked and keeps the five best.
  deep  The same, plus searches written by the language model from the question: two
        queries, and a third built on the model's own guess when it is sure of it. A
        model-written query is kept only if its first five documents match the question
        about as well as the two plain lists do; a document that only the guess query
        found must clearly beat the fifth document to enter the five.

The reader never sees the guess or the queries: only the question and the five passages.

Measured offline on the 258 test questions (Medline, November 2026): the right document is
among the five for 73 of 256 questions with fast against 41 with the old single search, and
the answer text for 205 of 258 against 147. Deep added 8 right answers of 258, which could
still be chance.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEPTHS = ("fast", "deep")
DEFAULT_DEPTH = "fast"

# What the person reading the answer is told when the service did less than asked.
NOTE_MESSAGES = {
    "depth_reduced": (
        "Our GPU is running slow at the moment, so the extra searches of the deep mode were "
        "skipped and this answer comes from the fast search. Try again in a moment for a "
        "deep search."
    ),
    "extractive_fallback": (
        "Our GPU is running slow at the moment, so this is the extractive answer: a passage "
        "copied from the best document. Try again in a moment for a generated answer."
    ),
    # Only on a server configured without the shared language model.
    "deep_unavailable": (
        "The deep mode is not available on this server, so this answer comes from the fast search."
    ),
}


def note(code: str) -> Dict[str, str]:
    return {"code": code, "message": NOTE_MESSAGES[code]}


@dataclass
class DepthSettings:
    candidates_per_search: int = 20
    subquery_timeout: float = 3.0         # no valid reply in this time: go on without
    extra_search_timeout: float = 1.0     # a model-written search slower than this counts as empty
    min_agreement: float = 0.8            # keep a model-written query whose first five match this well
    max_new_words: int = 5                # words a model-written query may add to the question's
    guess_query: bool = True              # search the model's guess when it says it is sure
    guess_margin: float = 1.0             # reranker logit a guess-only document needs above the fifth
    subquery_cache_ttl: float = 120.0
    batch_subquery_timeout: float = 30.0   # a batch job waits longer for the queries, at low priority     # the three tab calls of one question share the queries


# --------------------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------------------

_STOP = set(
    "a an the of in on at to for from by with and or is are was were be been which what who whom "
    "whose when where why how does do did has have had that this these those it its as into than "
    "not can could may might".split()
)


def toks(text: str) -> List[str]:
    """Content words: lower case, three letters or more, no stop word."""
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3 and t not in _STOP]


def new_words(question: str, text: str) -> List[str]:
    """Words of `text` that the question does not have (a five-letter stem counts as had)."""
    qt = set(toks(question))
    stems = {t[:5] for t in qt}
    return [t for t in toks(text) if not (t in qt or t[:5] in stems)]


def strip_guess(question: str, guess: str, query: str) -> str:
    """A word of the guess stays in a query only if the question has it too."""
    qt = set(toks(question))
    stems = {t[:5] for t in qt}
    g = set(toks(guess))
    keep = []
    for w in query.split():
        wt = toks(w)
        if wt and all(t in g for t in wt) and not all((t in qt) or (t[:5] in stems) for t in wt):
            continue
        keep.append(w)
    return " ".join(keep)


def clean_query(question: str, guess: str, query: str, max_new_words: int) -> Optional[str]:
    """The rules a model-written query must pass, applied in code: plain words only, the
    guess's words removed, 3 to 10 words, at most `max_new_words` words the question lacks.
    Returns None for a query that fails."""
    q = re.sub(r"[^\w\s\-\.]", " ", str(query or ""))
    q = re.sub(r"\b(?:AND|OR|NOT)\b", " ", q)
    q = " ".join(q.split())
    if guess:
        q = strip_guess(question, guess, q)
    words = q.split()
    if len(words) < 3:
        return None
    if len(words) > 10:
        words = words[:10]
        q = " ".join(words)
    if len(new_words(question, q)) > max_new_words:
        return None
    return q


# --------------------------------------------------------------------------------------
# Model-written queries
# --------------------------------------------------------------------------------------

SUBQ_SYSTEM = (
    "You prepare keyword searches for a BM25 search engine over PubMed titles and abstracts. "
    "The engine matches any of the words and ranks by word overlap; it knows no synonyms."
)

SUBQ_SCHEMA = {
    "type": "object",
    "properties": {
        "guess": {"type": "string"},
        "sure": {"type": "string", "enum": ["high", "medium", "low"]},
        "queries": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
            "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
    },
    "required": ["guess", "sure", "queries"],
}


def subquery_messages(question: str) -> List[dict]:
    return [
        {"role": "system", "content": SUBQ_SYSTEM},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            "Return JSON with three fields, in this order.\n"
            "guess: your own best guess at the answer, from memory, in at most six words; "
            "an empty string if you do not know.\n"
            "sure: how sure you are of the guess: high, medium or low.\n"
            "queries: two different search queries that together are likely to retrieve an "
            "abstract that answers the question.\n"
            "- Query 1: the same need with synonyms, alternative names or spellings, "
            "abbreviations expanded.\n"
            "- Query 2: a different angle: words an abstract that answers the question would "
            "use, whatever the answer is.\n"
            "Rules for the queries:\n"
            "- Each query: 3 to 10 plain keywords, no punctuation, no boolean operators, no "
            "quotation marks or brackets.\n"
            "- Keep species, gene, drug and disease names exactly as written in the question "
            "in every query.\n"
            "- No word of your guess may appear in a query, unless the question itself "
            "contains it. The guess has its own field."
        )},
    ]


@dataclass
class SubQueries:
    queries: List[str] = field(default_factory=list)
    guess: str = ""
    sure: Optional[str] = None
    guess_query: Optional[str] = None     # set only when the model is sure of its guess
    seconds: float = 0.0
    failed: Optional[str] = None          # why there are none: "slow", "busy", "error", ...


def parse_subqueries(question: str, text: str, max_new_words: int, with_guess_query: bool) -> SubQueries:
    try:
        j = json.loads(text)
    except (TypeError, ValueError):
        return SubQueries(failed="unreadable")
    if not isinstance(j, dict):
        return SubQueries(failed="unreadable")
    guess = " ".join(str(j.get("guess") or "").split())[:120]
    sure = j.get("sure") if j.get("sure") in ("high", "medium", "low") else None
    out: List[str] = []
    seen = {frozenset(toks(question))}  # a query that repeats the plain search adds nothing
    for item in j.get("queries") or []:
        raw = item.get("q") if isinstance(item, dict) else item
        q = clean_query(question, guess, raw, max_new_words)
        if q is None:
            continue
        key = frozenset(toks(q))
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    gq = None
    if with_guess_query and sure == "high" and guess and new_words(question, guess):
        gq = " ".join(dict.fromkeys(toks(question)[:10] + toks(guess)))
    return SubQueries(queries=out, guess=guess, sure=sure, guess_query=gq)


class SubQueryWriter:
    """Asks the language model for the queries, once per question.

    The web page sends one request per collection tab, three at once. They share one call:
    the first request writes the queries, the others wait for them, and the result is kept
    for `subquery_cache_ttl` seconds.
    """

    def __init__(self, chat: Callable[..., object], settings: DepthSettings, batch_priority: int = 10):
        self._chat = chat
        self.settings = settings
        self.batch_priority = batch_priority
        self._lock = threading.Lock()
        self._done: Dict[str, Tuple[float, SubQueries]] = {}
        self._running: Dict[str, threading.Event] = {}

    def _limit(self, batch: bool) -> float:
        return self.settings.batch_subquery_timeout if batch else self.settings.subquery_timeout

    def _write(self, question: str, batch: bool = False) -> SubQueries:
        from .llm_remote import LLMUnavailable
        t0 = time.perf_counter()
        limit = self._limit(batch)
        try:
            comp = self._chat(
                subquery_messages(question), max_tokens=200, temperature=0.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, presence_penalty=0.0, seed=0,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "subqueries", "schema": SUBQ_SCHEMA}},
                first_token_timeout=limit, total_timeout=limit,
                priority=self.batch_priority if batch else None,
            )
        except LLMUnavailable as e:
            return SubQueries(failed=e.reason, seconds=time.perf_counter() - t0)
        sq = parse_subqueries(question, comp.text, self.settings.max_new_words, self.settings.guess_query)
        sq.seconds = time.perf_counter() - t0
        return sq

    def get(self, question: str, batch: bool = False) -> SubQueries:
        # A batch job's call (low priority, long limit) is not shared with people's requests.
        key = ("batch|" if batch else "") + " ".join(question.split()).lower()
        with self._lock:
            now = time.monotonic()
            for k in [k for k, (ts, _) in self._done.items() if now - ts > self.settings.subquery_cache_ttl]:
                del self._done[k]
            if key in self._done:
                return self._done[key][1]
            event = self._running.get(key)
            runner = event is None
            if runner:
                event = threading.Event()
                self._running[key] = event
        if not runner:
            # As long as the call itself may take: the time limit plus connecting.
            event.wait(self._limit(batch) + 3.0)
            with self._lock:
                hit = self._done.get(key)
            return hit[1] if hit else SubQueries(failed="slow")
        try:
            sq = self._write(question, batch)
        except Exception as e:  # never let the optional step break the request
            sq = SubQueries(failed=f"error: {type(e).__name__}")
        with self._lock:
            if not sq.failed:  # a failure is not kept: the next request tries again
                self._done[key] = (time.monotonic(), sq)
            self._running.pop(key, None)
        event.set()
        return sq


# --------------------------------------------------------------------------------------
# Choosing the five
# --------------------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def doc_key(doc) -> str:
    return f"{getattr(doc, 'source', '')}:{getattr(doc, 'doc_id', '')}"


def merge(*lists: Sequence) -> List:
    """Documents of all lists, first occurrence kept, in list order."""
    out, seen = [], set()
    for lst in lists:
        for d in lst or []:
            k = doc_key(d)
            if k not in seen:
                seen.add(k)
                out.append(d)
    return out


@dataclass
class Selection:
    documents: List                     # the five (or fewer), best first
    scores: List[float]                 # their reranker logits
    n_candidates: int
    kept_queries: List[str]
    dropped_queries: List[str]
    guess_docs_in: int = 0


def select(
    question: str,
    base_lists: Sequence[Sequence],
    extra: Sequence[Tuple[str, Sequence]],
    guess_list: Optional[Sequence],
    score: Callable[[str, List], List[float]],
    final_n: int,
    settings: DepthSettings,
) -> Selection:
    """Rerank the candidates against the question as asked and keep `final_n`.

    `base_lists`: today's list and the plain OR list. `extra`: (query, documents) of each
    model-written query. `guess_list`: the documents of the guess query, or None.
    `score(question, docs)` returns one reranker logit per document.
    """
    logit: Dict[str, float] = {}

    def score_new(docs):
        todo = [d for d in docs if doc_key(d) not in logit]
        if todo:
            for d, s in zip(todo, score(question, todo)):
                logit[doc_key(d)] = float(s)

    pool = merge(*base_lists)
    # First pass: the two plain lists, and the first five of every model-written list,
    # which is all the agreement test needs.
    score_new(pool + [d for _, docs in extra for d in list(docs)[:5]]
              + (list(guess_list)[:5] if guess_list else []))

    def mean5(docs):
        top = [d for d in list(docs)[:5] if doc_key(d) in logit]
        return sum(sigmoid(logit[doc_key(d)]) for d in top) / 5 if top else 0.0

    ref = max([mean5(lst) for lst in base_lists] + [1e-6])
    kept, dropped, kept_docs = [], [], []
    for q, docs in extra:
        agreement = min(1.5, mean5(docs) / ref)
        if docs and agreement >= settings.min_agreement:
            kept.append(q)
            kept_docs.extend(docs)
        else:
            dropped.append(q)

    # Second pass: the rest of the kept lists, and the guess list.
    candidates = merge(pool, kept_docs)
    guess_only = [d for d in (guess_list or []) if doc_key(d) not in {doc_key(c) for c in candidates}]
    score_new(candidates + guess_only)

    ranked = sorted(candidates, key=lambda d: logit[doc_key(d)], reverse=True)
    chosen = ranked[:final_n]
    guess_in = 0
    if guess_only:
        floor = logit[doc_key(chosen[-1])] if len(chosen) >= final_n else -math.inf
        for d in sorted(guess_only, key=lambda d: logit[doc_key(d)], reverse=True):
            if logit[doc_key(d)] > floor + settings.guess_margin:
                chosen.append(d)
                guess_in += 1
        chosen = sorted(chosen, key=lambda d: logit[doc_key(d)], reverse=True)[:final_n]
        guess_keys = {doc_key(d) for d in guess_only}
        guess_in = sum(1 for d in chosen if doc_key(d) in guess_keys)
    return Selection(
        documents=chosen,
        scores=[logit[doc_key(d)] for d in chosen],
        n_candidates=len(candidates) + len(guess_only),
        kept_queries=kept,
        dropped_queries=dropped,
        guess_docs_in=guess_in,
    )
