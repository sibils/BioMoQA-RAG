"""The depth logic, with fake documents and a fake reranker: no network, no model."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from biomoqa_rag import depth as D
from biomoqa_rag.llm_remote import LLMUnavailable


def doc(i, source="medline"):
    return SimpleNamespace(doc_id=str(i), source=source, title=f"t{i}", abstract=f"a{i}", score=0.0)


Q = "What is the body length of Folsomia candida?"


# --- cleaning the model's queries ------------------------------------------------------

def test_clean_query_keeps_plain_words():
    assert D.clean_query(Q, "", "Folsomia candida body size mm", 5) == "Folsomia candida body size mm"


def test_clean_query_drops_punctuation_and_boolean_words():
    assert D.clean_query(Q, "", '"Folsomia candida" AND (length OR size)', 5) == "Folsomia candida length size"


def test_clean_query_needs_three_words():
    assert D.clean_query(Q, "", "Folsomia candida", 5) is None


def test_clean_query_cuts_at_ten_words():
    q = "Folsomia candida body length adult juvenile size measurement collembola springtail extra"
    assert len(D.clean_query(Q, "", q, 9).split()) == 10


def test_clean_query_rejects_more_than_five_new_words():
    q = "Folsomia candida alpha beta gamma delta epsilon zeta"
    assert D.clean_query(Q, "", q, 5) is None
    assert D.clean_query(Q, "", "Folsomia candida alpha beta gamma delta epsilon", 5) is not None


def test_guess_words_are_removed_unless_the_question_has_them():
    assert D.clean_query(Q, "about 2.5 millimetres", "Folsomia candida length millimetres", 5) == \
        "Folsomia candida length"
    # "length" is in the question: kept even though the guess has it
    assert D.clean_query(Q, "length 2 mm", "Folsomia candida length adult", 5) == "Folsomia candida length adult"


# --- parsing the reply -----------------------------------------------------------------

def reply(guess="", sure="low", queries=("Folsomia candida size adults", "springtail measured length adults")):
    return json.dumps({"guess": guess, "sure": sure, "queries": [{"q": q} for q in queries]})


def test_parse_subqueries_reads_two_queries():
    sq = D.parse_subqueries(Q, reply(), 5, True)
    assert sq.queries == ["Folsomia candida size adults", "springtail measured length adults"]
    assert sq.guess_query is None and not sq.failed


def test_parse_subqueries_drops_a_query_that_repeats_the_question():
    sq = D.parse_subqueries(Q, reply(queries=("body length Folsomia candida", "springtail measured length adults")), 5, True)
    assert sq.queries == ["springtail measured length adults"]


def test_guess_query_only_when_sure():
    assert D.parse_subqueries(Q, reply(guess="2 mm", sure="medium"), 5, True).guess_query is None
    gq = D.parse_subqueries(Q, reply(guess="about 2 millimetres", sure="high"), 5, True).guess_query
    assert gq and "millimetres" in gq and "folsomia" in gq
    assert D.parse_subqueries(Q, reply(guess="about 2 millimetres", sure="high"), 5, False).guess_query is None


def test_unreadable_reply_is_a_failure():
    assert D.parse_subqueries(Q, "not json", 5, True).failed == "unreadable"


# --- choosing the five -----------------------------------------------------------------

class Scorer:
    """Logit per document id, and a record of how many documents were scored."""

    def __init__(self, logits):
        self.logits = logits
        self.scored = []

    def __call__(self, question, docs):
        self.scored.extend(d.doc_id for d in docs)
        return [self.logits.get(d.doc_id, -5.0) for d in docs]


def test_select_reranks_the_merged_lists():
    a, b = [doc(1), doc(2), doc(3)], [doc(3), doc(4), doc(5), doc(6)]
    sc = Scorer({"1": 0.0, "2": 1.0, "3": 3.0, "4": -1.0, "5": 2.0, "6": 4.0})
    sel = D.select(Q, [a, b], [], None, sc, 5, D.DepthSettings())
    assert [d.doc_id for d in sel.documents] == ["6", "3", "5", "2", "1"]
    assert sel.n_candidates == 6
    assert sorted(sc.scored) == ["1", "2", "3", "4", "5", "6"]  # each document scored once


def test_select_keeps_a_model_query_only_when_it_matches_the_question():
    a, b = [doc(i) for i in range(1, 6)], []
    good = [doc(i) for i in range(10, 16)]
    bad = [doc(i) for i in range(20, 26)]
    logits = {str(i): 2.0 for i in range(1, 6)}
    logits.update({str(i): 2.5 for i in range(10, 16)})
    logits.update({str(i): -3.0 for i in range(20, 26)})
    sc = Scorer(logits)
    sel = D.select(Q, [a, b], [("good query", good), ("bad query", bad)], None, sc, 5, D.DepthSettings())
    assert sel.kept_queries == ["good query"] and sel.dropped_queries == ["bad query"]
    assert all(d.doc_id in {str(i) for i in range(10, 16)} for d in sel.documents)
    # only the first five of the dropped list were scored
    assert len([i for i in sc.scored if int(i) >= 20]) == 5


def test_guess_documents_need_a_margin_over_the_fifth():
    base = [doc(i) for i in range(1, 6)]
    logits = {str(i): float(i) for i in range(1, 6)}   # the fifth (lowest kept) is 1.0
    logits.update({"30": 1.5, "31": 2.5})             # 31 beats it by more than 1.0, 30 does not
    sel = D.select(Q, [base, []], [], [doc(30), doc(31)], Scorer(logits), 5, D.DepthSettings(guess_margin=1.0))
    ids = [d.doc_id for d in sel.documents]
    assert "31" in ids and "30" not in ids and sel.guess_docs_in == 1 and len(ids) == 5


def test_select_with_no_plain_results_keeps_model_lists():
    extra = [doc(i) for i in range(10, 13)]
    sel = D.select(Q, [[], []], [("q", extra)], None, Scorer({}), 5, D.DepthSettings())
    assert sel.kept_queries == ["q"] and len(sel.documents) == 3


def test_documents_from_different_collections_do_not_collide():
    a = [doc(1, "medline")]
    b = [doc(1, "pmc")]
    sel = D.select(Q, [a, b], [], None, Scorer({}), 5, D.DepthSettings())
    assert len(sel.documents) == 2


# --- the writer: one call per question -------------------------------------------------

def test_writer_shares_one_call_between_concurrent_requests():
    calls = []

    def chat(messages, **kw):
        calls.append(kw)
        time.sleep(0.2)
        return SimpleNamespace(text=reply())

    w = D.SubQueryWriter(chat, D.DepthSettings())
    out = []
    threads = [threading.Thread(target=lambda: out.append(w.get(Q))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(out) == 3 and all(o.queries for o in out)
    assert calls[0]["first_token_timeout"] == 3.0 and calls[0]["total_timeout"] == 3.0
    w.get(Q)
    assert len(calls) == 1  # kept for the next tab


def test_writer_failure_is_reported_and_not_kept():
    n = []

    def chat(messages, **kw):
        n.append(1)
        raise LLMUnavailable("slow", "no first token")

    w = D.SubQueryWriter(chat, D.DepthSettings())
    assert w.get(Q).failed == "slow"
    assert w.get(Q).failed == "slow"
    assert len(n) == 2


def test_a_batch_job_writes_its_own_queries_at_low_priority_and_may_wait():
    calls = []

    def chat(messages, **kw):
        calls.append(kw)
        return SimpleNamespace(text=reply())

    w = D.SubQueryWriter(chat, D.DepthSettings(), batch_priority=10)
    w.get(Q)                 # a person's request
    w.get(Q, batch=True)     # a script's: not served from the person's call, nor the reverse
    w.get(Q, batch=True)
    assert len(calls) == 2
    assert calls[0]["priority"] is None and calls[0]["first_token_timeout"] == 3.0
    assert calls[1]["priority"] == 10 and calls[1]["first_token_timeout"] == 30.0


def test_every_note_has_a_message():
    for code in ("depth_reduced", "extractive_fallback", "deep_unavailable"):
        nt = D.note(code)
        assert nt["code"] == code and "moment" in nt["message"] or code == "deep_unavailable"
