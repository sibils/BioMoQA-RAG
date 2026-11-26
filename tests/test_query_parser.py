"""The query parser's memory: bounded, expiring, never shared with callers, failures not kept."""
from types import SimpleNamespace

import pytest
import requests

from biomoqa_rag.retrieval import query_parser as QP


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


@pytest.fixture
def calls(monkeypatch):
    """Every question the fake parser service was asked, in order."""
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append(params["query"])
        if params["query"].startswith("bad"):
            return FakeResponse({}, status=400)
        col = params.get("es")
        es = {col: {"query": {"bool": {"must": [{"match": {"text": params["query"]}}]}}}} if col else {}
        return FakeResponse({"success": True, "normalized_query": params["query"], "es_queries": es,
                             "text_parts": params["query"].split()})

    monkeypatch.setattr(QP.requests, "get", fake_get)
    return seen


def test_a_repeated_question_is_parsed_once(calls):
    p = QP.SIBILSQueryParser()
    assert p.parse("a b").success and p.parse("a b").success
    assert calls == ["a b"]


def test_the_least_recently_used_parse_goes_first(calls):
    p = QP.SIBILSQueryParser(cache_size=2)
    for q in ("a", "b", "a", "c"):   # "a" used again, so "c" pushes out "b"
        p.parse(q)
    p.parse("a")
    p.parse("b")
    assert calls == ["a", "b", "c", "b"] and len(p._cache) == 2


def test_an_old_parse_is_asked_for_again(calls, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(QP, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    p = QP.SIBILSQueryParser(cache_ttl=60)
    p.parse("a")
    clock[0] += 59
    p.parse("a")
    clock[0] += 2
    p.parse("a")
    assert calls == ["a", "a"]


def test_a_failed_parse_is_not_kept(calls):
    p = QP.SIBILSQueryParser()
    assert not p.parse("bad question").success
    p.parse("bad question")
    assert calls == ["bad question", "bad question"] and not p._cache


def test_each_caller_gets_its_own_copy(calls):
    p = QP.SIBILSQueryParser()
    first = p.parse("a")
    first.es_query["query"]["bool"]["filter"] = ["added by a caller"]
    assert "filter" not in p.parse("a").es_query["query"]["bool"]


def test_collection_and_options_are_part_of_the_key(calls):
    p = QP.SIBILSQueryParser()
    p.parse("a", collection="medline")
    p.parse("a", collection="plazi")
    p.parse("a", collection="plazi", include_es_query=False)
    p.parse("a", collection="plazi")
    assert calls == ["a", "a", "a"]
