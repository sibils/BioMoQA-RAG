"""The SIBILS retriever's query building, with the network replaced: no request is sent."""
import json
from types import SimpleNamespace

import pytest

from biomoqa_rag.retrieval import sibils_retriever as S


def retriever(tmp_path, parser=None):
    r = S.SIBILSRetriever(collection=["medline"], cache_dir=str(tmp_path), use_query_parser=parser is not None)
    if parser is not None:
        r.query_parser = parser
    return r


class FakeParser:
    """Rejects any question with a bracket, like the real parser (HTTP 400)."""

    def __init__(self):
        self.seen = []

    def parse(self, text, collection=None, include_es_query=True):
        self.seen.append(text)
        if "(" in text or "[" in text:
            return SimpleNamespace(success=False, es_query=None, text_parts=None)
        es = {"query": {"bool": {"must": [{"term": {"annotations.all": "ott|123|Folsomia candida"}},
                                          {"term": {"annotations.all": "ott|9|Scientific name"}}]}}}
        return SimpleNamespace(success=True, es_query=es, text_parts=text.split())


def test_parser_rejection_is_retried_without_brackets(tmp_path):
    p = FakeParser()
    r = retriever(tmp_path, p)
    parsed, taxa, text = r.parse("Is adriamycin(doxorubicin) toxic to Folsomia candida?", "medline")
    assert parsed.success and text == "Is adriamycin doxorubicin toxic to Folsomia candida?"
    assert len(p.seen) == 2
    assert taxa == [("Folsomia candida", "ott|123|Folsomia candida")]  # the junk name is dropped


def test_species_filter_asks_for_any_of_several(tmp_path):
    r = retriever(tmp_path)
    f = r._taxon_filter([("Folsomia candida", "ott|1|Folsomia candida"), ("Isotoma viridis", "")], "plazi")
    assert f["bool"]["minimum_should_match"] == 1
    phrases = [c["multi_match"]["query"] for c in f["bool"]["should"] if "multi_match" in c]
    assert phrases == ["Folsomia candida", "Isotoma viridis"]
    assert f["bool"]["should"][0]["multi_match"]["fields"] == ["treatment_title", "text", "title"]
    q = r._with_taxon_phrase({"query": {"match": {"title": "x"}}}, [("Folsomia candida", "")], "medline")
    assert len(q["query"]["bool"]["filter"]) == 1


class FakeResponse:
    def __init__(self, hits):
        self.hits = hits

    def raise_for_status(self):
        pass

    def json(self):
        return {"success": True, "elastic_output": {"hits": {"hits": self.hits}}}


def hit(i, abstract="An abstract with words."):
    return {"_id": str(i), "_score": 1.0, "_source": {"title": f"T{i}", "abstract": abstract}}


def test_plain_search_query_and_cache(tmp_path, monkeypatch):
    sent = []

    def post(url, params=None, data=None, timeout=None):
        sent.append((params, json.loads(data["jq"]), timeout))
        return FakeResponse([hit(1), hit(2, abstract="")])  # the title-only citation is dropped

    monkeypatch.setattr(S.requests, "post", post)
    r = retriever(tmp_path)
    docs = r.plain_search("What is the size of Folsomia candida?", "medline", 20, [("Folsomia candida", "")])
    assert [d.doc_id for d in docs] == ["1"]
    params, jq, timeout = sent[0]
    assert params == {"col": "medline", "n": 20}
    mm = jq["query"]["bool"]["must"][0]["multi_match"]
    assert mm["operator"] == "or" and mm["query"] == "What is the size of Folsomia candida"
    assert "filter" in jq["query"]["bool"]
    r.plain_search("What is the size of Folsomia candida?", "medline", 20, [("Folsomia candida", "")])
    assert len(sent) == 1  # second call from the cache
    r.plain_search("What is the size of Folsomia candida?", "medline", 20, [])
    assert len(sent) == 2  # another species filter is another cache entry


def test_optional_search_failure_is_an_empty_list(tmp_path, monkeypatch):
    def post(*a, **k):
        raise S.requests.exceptions.ReadTimeout("slow")

    monkeypatch.setattr(S.requests, "post", post)
    r = retriever(tmp_path)
    assert r.plain_search("some words here", "medline", 20, [], timeout=1.0) == []
    with pytest.raises(Exception):
        r.plain_search("some words here", "medline", 20, [])


def test_today_list_has_no_wildcard_filter(tmp_path, monkeypatch):
    sent = []

    def post(url, params=None, data=None, timeout=None):
        sent.append(json.loads(data["jq"]))
        return FakeResponse([hit(1)])

    monkeypatch.setattr(S.requests, "post", post)
    r = retriever(tmp_path, FakeParser())
    docs = r.today_list("Body length of Folsomia candida?", "medline", 20)
    assert [d.doc_id for d in docs] == ["1"]
    assert "wildcard" not in json.dumps(sent[0])
    assert len(sent[0]["query"]["bool"]["filter"]) == 1  # the species filter only


def test_cache_keys_carry_the_version_and_the_kind(tmp_path):
    r = retriever(tmp_path)
    assert r._cache_key("q", "medline", 20, "today") != r._cache_key("q", "medline", 20, "or|")
    old = S.SIBILSRetriever.CACHE_VERSION
    k1 = r._cache_key("q", "medline", 20)
    S.SIBILSRetriever.CACHE_VERSION = "older"
    try:
        assert r._cache_key("q", "medline", 20) != k1
    finally:
        S.SIBILSRetriever.CACHE_VERSION = old
