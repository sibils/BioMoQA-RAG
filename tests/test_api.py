"""The HTTP API with a fake pipeline: the depth parameter, the new fields, the logs."""
import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from biomoqa_rag import api_server as A
from biomoqa_rag.llm_remote import LLMUnavailable
from biomoqa_rag.retrieval.sibils_retriever import Document

BROWSER = {"Sec-Fetch-Mode": "cors", "Origin": "https://www.sibils.org"}   # what a browser adds

Q = "What is the body length of Folsomia candida?"


class FakePipeline:
    def __init__(self):
        self.config = A.RAGConfig()
        self.calls = []
        self.reading, self.most_reading = 0, 0
        self.lock = threading.Lock()
        self.notes, self.fail_batch = [], False

    def run_with_documents(self, question, documents, mode, **kw):
        with self.lock:
            self.reading += 1
            self.most_reading = max(self.most_reading, self.reading)
        time.sleep(0.3)                       # one answer's time
        with self.lock:
            self.reading -= 1
        col = documents[0].source
        return {"answers": [{"answer": f"Adults are 2 mm ({col}).", "answer_score": None,
                             "docs": [{"docid": documents[0].doc_id, "doc_retrieval_score": 1.0}]}],
                "mode_used": "extractive" if col == "pmc" else mode, "notes": []}

    def run(self, **kw):
        self.calls.append(kw)
        return {
            "sibils_version": "biomoqa-2.0", "success": True, "error": "", "question": kw["question"],
            "collection": kw["collection"], "model": "biobert", "ndocs_requested": kw["final_n"] or 5,
            "ndocs_returned_by_SIBiLS": 30,
            "answers": [{"answer": "2 mm", "answer_score": 0.9, "docs": [{"docid": "1", "doc_source": "medline"}]}],
            "mode_used": "extractive", "depth_used": "fast",
            "notes": [{"code": "extractive_fallback", "message": "Our GPU is running slow at the moment ..."}],
            "queries_used": [], "pipeline_time": 1.2, "transformed_query": None, "_timings": {"rerank": 0.4},
        }

    def run_multi_collection(self, **kw):
        self.calls.append(kw)
        if self.fail_batch and kw.get("batch"):
            raise LLMUnavailable("slow", "no first token in 60 s")
        return {
            "question": kw["question"], "mode_used": "generative", "ndocs_retrieved": 60, "model": "qwen3.8-27b",
            "pipeline_time": 4.0, "_timings": {}, "_notes": list(self.notes), "_depth_used": ["deep"],
            "collection_results": [{"collection": "plazi", "rank": 1, "mode_used": "generative",
                                    "depth_used": "deep", "notes": [], "queries_used": ["Folsomia candida size"],
                                    "answers": [{"answer": "Adults are 2 mm.", "docs": []}]}],
        }


@pytest.fixture()
def client(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(A, "pipeline", fake)
    A._multi_cache.clear()
    return TestClient(A.app), fake


def test_get_qa_passes_depth_and_returns_the_new_fields(client):
    c, fake = client
    r = c.get("/QA", params={"q": Q, "col": "medline", "mode": "generative", "n": 50, "depth": "deep",
                             "retrieval": "dense"})
    assert r.status_code == 200
    body = r.json()
    assert body["depth_used"] == "fast" and body["notes"][0]["code"] == "extractive_fallback"
    assert body["queries_used"] == [] and "_timings" not in body
    assert fake.calls[0]["depth"] == "deep" and fake.calls[0]["final_n"] == 5  # n is capped at 5


def test_unknown_depth_is_a_422(client):
    c, _ = client
    assert c.get("/QA", params={"q": Q, "col": "medline", "depth": "deepest"}).status_code == 422
    assert c.post("/qa", json={"question": Q, "depth": "deepest"}).status_code == 422


def test_post_qa_takes_depth_in_the_body(client):
    c, fake = client
    r = c.post("/qa", json={"question": Q, "mode": "generative", "depth": "deep", "retrieval": "dense"})
    assert r.status_code == 200
    res = r.json()["collection_results"][0]
    assert res["depth_used"] == "deep" and res["queries_used"] == ["Folsomia candida size"]
    assert fake.calls[0]["depth"] == "deep"


class FakeResolver:
    """doc_refs as the trait pipeline sends them: one document in each collection."""
    def resolve_batch(self, refs):
        cols = ["medline", "plazi", "pmc"]
        return [Document(doc_id=r, title="t", abstract="Adults are 2 mm.", source=c) for r, c in zip(refs, cols)], []


def test_doc_refs_reads_the_collections_at_once_and_names_what_answered(client, monkeypatch):
    c, fake = client
    monkeypatch.setattr(A, "get_doc_resolver", lambda: FakeResolver())
    t0 = time.perf_counter()
    body = c.post("/qa", json={"question": Q, "mode": "generative", "doc_refs": ["1", "2", "3"]}).json()
    assert fake.most_reading == 3 and time.perf_counter() - t0 < 0.8      # not 3 x 0.3 s in a row
    by_col = {r["collection"]: r for r in body["collection_results"]}
    assert by_col["pmc"]["mode_used"] == "extractive" and by_col["plazi"]["mode_used"] == "generative"
    top = body["collection_results"][0]["mode_used"]
    # the shared server's model, not the local one config.toml still names ([model] model_name)
    assert body["mode_used"] == top
    assert body["model"] == ("qwen3.8-27b" if top == "generative" else "biobert")
    assert "8b" not in body["model"]


def test_a_scripts_post_is_a_batch_job_and_a_browsers_is_not(client):
    c, fake = client
    c.post("/qa", json={"question": Q})                                  # a script (no browser headers)
    c.post("/qa", json={"question": Q + " "}, headers=BROWSER)           # the web page
    c.post("/qa/multi", json={"question": Q})
    c.get("/QA", params={"q": Q, "col": "medline"})                       # GET: never a batch job
    assert [k.get("batch") for k in fake.calls] == [True, False, True, None]


def test_doc_refs_from_a_script_are_a_batch_job(client, monkeypatch):
    c, fake = client
    seen = []
    real = fake.run_with_documents
    monkeypatch.setattr(fake, "run_with_documents", lambda *a, **kw: seen.append(kw.get("batch")) or real(*a, **kw))
    monkeypatch.setattr(A, "get_doc_resolver", lambda: FakeResolver())
    c.post("/qa", json={"question": Q, "mode": "generative", "doc_refs": ["1"]})
    c.post("/qa", json={"question": Q, "mode": "generative", "doc_refs": ["1"]}, headers=BROWSER)
    assert seen == [True, False]


def test_batch_jobs_and_people_do_not_share_cached_answers(client):
    c, fake = client
    c.post("/qa", json={"question": Q}, headers=BROWSER)
    c.post("/qa", json={"question": Q})                  # a script: its own run, not the page's answer
    c.post("/qa", json={"question": Q})                  # the same script again: from cache
    assert len(fake.calls) == 2


def test_an_answer_with_a_fallback_note_is_not_kept_for_two_minutes(client, monkeypatch):
    c, fake = client
    fake.notes = ["extractive_fallback"]
    monkeypatch.setattr(A, "_DEGRADED_TTL", 0)           # "try again in a moment" must run again
    c.post("/qa", json={"question": Q}, headers=BROWSER)
    c.post("/qa", json={"question": Q}, headers=BROWSER)
    assert len(fake.calls) == 2
    fake.notes = []
    c.post("/qa", json={"question": Q + "?"}, headers=BROWSER)
    c.post("/qa", json={"question": Q + "?"}, headers=BROWSER)
    assert len(fake.calls) == 3                          # a full answer stays cached


def test_a_batch_job_the_model_cannot_answer_gets_an_error_not_biobert(client):
    c, fake = client
    fake.fail_batch = True
    body = c.post("/qa", json={"question": Q, "mode": "generative"}).json()
    assert body["collection_results"] == [] and "no first token in 60 s" in body["error"]


def test_health_reports_the_commit(client):
    c, _ = client
    body = c.get("/health").json()
    assert "commit" in body and body["features"]["depths"] == ["fast", "deep"]
    assert body["config"]["generation_backend"] == "remote"


def test_logs_have_times_and_codes_but_not_the_question(client, caplog):
    c, _ = client
    with caplog.at_level(logging.INFO, logger="biomoqa"):
        c.get("/QA", params={"q": Q, "col": "medline"})
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("qa_request")]
    assert lines and "extractive_fallback" in lines[0] and "rerank" in lines[0]
    assert "Folsomia" not in lines[0]


def test_access_log_drops_the_query_string():
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4", "GET", "/api/QA?q=secret+question&col=medline", "1.1", 200), None)
    A._NoQueryString().filter(rec)
    assert "secret" not in rec.getMessage() and "/api/QA" in rec.getMessage()
