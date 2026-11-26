"""The pipeline's depth search, reading and fallback, with every service faked: no model loaded."""
from types import SimpleNamespace

import pytest

from biomoqa_rag import pipeline as P
from biomoqa_rag.depth import DepthSettings, SubQueries
from biomoqa_rag.llm_remote import Completion, LLMUnavailable
from biomoqa_rag.retrieval.sibils_retriever import Document

Q = "What is the body length of Folsomia candida?"


def d(i, source="medline", abstract=None):
    return Document(doc_id=str(i), title=f"Title {i}", abstract=abstract or f"Folsomia candida adults measure {i} mm.",
                    source=source, pmid=str(i))


class FakeSibils:
    collection = ["medline", "plazi", "pmc"]

    def __init__(self):
        self.plain_calls = []

    def parse(self, question, collection):
        return (None, [("Folsomia candida", "")], question)

    def today_list(self, question, collection, n, parsed=None):
        return [d(1, collection), d(2, collection)]

    def plain_search(self, text, collection, n, taxa, timeout=None):
        self.plain_calls.append((text, timeout))
        if text == Q:
            return [d(2, collection), d(3, collection)]
        k = len(self.plain_calls)
        return [d(10 * k + j, collection) for j in range(5)]   # five on-topic documents


class FakeReranker:
    def score(self, question, docs):
        return [float(int(x.doc_id)) for x in docs]   # a higher id scores higher


class FakeExtractor:
    def extract(self, question, documents, max_len):
        return [{"text": "2 mm", "score": 0.9, "doc_idx": 0, "span_start": 0, "span_end": 4,
                 "passage": documents[0].abstract}]


class FakeRemote:
    def __init__(self, fail=None, texts=("Adults are 2 mm long [0].",)):
        self.fail, self.texts, self.calls = fail, list(texts), []

    def chat(self, messages, **kw):
        self.calls.append(kw)
        if self.fail:
            raise LLMUnavailable(self.fail, "test")
        return Completion(text=self.texts.pop(0) if self.texts else "x", ttft=0.1, total=0.5, logprobs=[-0.1])


class FakeWriter:
    def __init__(self, sq):
        self.sq, self.batch = sq, []

    def get(self, question, batch=False):
        self.batch.append(batch)
        return self.sq


def pipe(remote=None, writer=None):
    p = object.__new__(P.RAGPipeline)
    p.config = P.RAGConfig(depth=DepthSettings(guess_margin=0.5))
    p.sibils = FakeSibils()
    p.reranker = FakeReranker()
    p.remote = remote or FakeRemote()
    p.subqueries = writer
    p._extractor = FakeExtractor()
    p._generation_lock = None
    return p


def test_fast_reranks_both_lists():
    p = pipe()
    docs, n, info = p._retrieve_collection(Q, "medline", "fast", 5)
    assert [x.doc_id for x in docs] == ["3", "2", "1"] and n == 3
    assert info["depth_used"] == "fast" and info["notes"] == [] and info["queries_used"] == []
    assert 0.9 < docs[0].score <= 1.0   # the reranker's probability


def test_deep_adds_the_model_searches():
    sq = SubQueries(queries=["Folsomia candida size adults", "springtail length measured"], guess="", sure="low")
    p = pipe(writer=FakeWriter(sq))
    docs, n, info = p._retrieve_collection(Q, "medline", "deep", 5)
    assert info["depth_used"] == "deep" and info["notes"] == []
    assert set(info["queries_used"]) <= set(sq.queries) and info["queries_used"]
    assert [t for t, timeout in p.sibils.plain_calls if timeout == 1.0] == sq.queries


def test_deep_falls_back_to_fast_when_the_model_is_slow():
    p = pipe(writer=FakeWriter(SubQueries(failed="slow")))
    docs, n, info = p._retrieve_collection(Q, "medline", "deep", 5)
    assert info["depth_used"] == "fast" and [x["code"] for x in info["notes"]] == ["depth_reduced"]


def test_generative_answer():
    p = pipe()
    notes = []
    answers, mode = p._read(Q, [d(1), d(2)], "generative", notes)
    assert mode == "generative" and notes == []
    assert answers[0]["answer"] == "Based on the provided documents, Adults are 2 mm long."
    assert answers[0]["docs"][0]["docid"] == "1"
    kw = p.remote.calls[0]
    assert kw["prefix"] == P.ANSWER_PREFIX and kw["repetition_penalty"] == 1.0 and kw["seed"] == 0


def test_citation_markers_leave_no_stray_punctuation():
    p = pipe(remote=FakeRemote(texts=(
        "the host is *Lissemys punctata* [0], found in the kidneys [1], [2]. "
        "Adults [0, 3] take up [18F]altanserin [4].",)))
    answers, _ = p._read(Q, [d(i) for i in range(1, 6)], "generative", [])
    assert answers[0]["answer"] == ("Based on the provided documents, the host is *Lissemys punctata*, "
                                    "found in the kidneys. Adults take up [18F]altanserin.")
    assert [x["docid"] for x in answers[0]["docs"]] == ["1", "2", "3", "4", "5"]


def test_blank_answer_is_asked_once_more():
    p = pipe(remote=FakeRemote(texts=("   ", "Adults are 2 mm [0].")))
    answers, mode = p._read(Q, [d(1)], "generative", [])
    assert len(p.remote.calls) == 2 and p.remote.calls[1]["seed"] == 1 and mode == "generative"


def test_slow_model_gives_the_extractive_answer_with_a_note():
    p = pipe(remote=FakeRemote(fail="slow"))
    notes = []
    answers, mode = p._read(Q, [d(1), d(2)], "generative", notes)
    assert mode == "extractive" and answers[0]["answer"] == "2 mm"
    assert [x["code"] for x in notes] == ["extractive_fallback"]
    assert "GPU" in notes[0]["message"] and "Try again in a moment" in notes[0]["message"]


def test_a_batch_job_waits_at_low_priority_and_never_gets_biobert_instead():
    p = pipe(remote=FakeRemote(fail="slow"))
    with pytest.raises(LLMUnavailable):
        p._read(Q, [d(1)], "generative", [], batch=True)
    p = pipe()
    answers, mode = p._read(Q, [d(1)], "generative", [], batch=True)
    kw = p.remote.calls[0]
    assert mode == "generative" and kw["priority"] == 10
    assert kw["first_token_timeout"] == 60.0 and kw["total_timeout"] == 100.0
    p = pipe()
    p._read(Q, [d(1)], "generative", [])                 # a person's request: the defaults
    assert "priority" not in p.remote.calls[0] and "first_token_timeout" not in p.remote.calls[0]


def test_batch_reaches_the_deep_queries():
    writer = FakeWriter(SubQueries(queries=["Folsomia candida size adults"], guess="", sure="low"))
    p = pipe(writer=writer)
    p.run_multi_collection(Q, mode="generative", depth="deep", batch=True)
    assert writer.batch and all(writer.batch)


def test_run_reports_what_ran():
    sq = SubQueries(queries=["Folsomia candida size adults"], guess="", sure="low")
    p = pipe(remote=FakeRemote(fail="busy"), writer=FakeWriter(sq))
    r = p.run(Q, collection="medline", mode="generative", depth="deep", final_n=3)
    assert r["depth_used"] == "deep" and r["mode_used"] == "extractive" and r["model"] == "biobert"
    assert [x["code"] for x in r["notes"]] == ["extractive_fallback"]
    assert r["ndocs_requested"] == 3 and len(r["answers"]) == 1 and "_timings" in r


def test_unknown_depth_is_refused():
    import pytest
    with pytest.raises(ValueError):
        pipe().run(Q, collection="medline", depth="deepest")


def test_short_documents_are_given_whole_and_long_ones_a_window():
    p = pipe()
    short = d(1, abstract="Short abstract. " * 100)            # 1,600 characters
    long_ = d(2, abstract="Filler text here. " * 300 + "Folsomia candida body length 2 mm. " + "More. " * 50)
    out = p._passages(Q, [short, long_])
    assert len(out[0]) >= 1500 and len(out[1]) <= p.config.llm_abstract_length


def test_langgraph_sandbox_searches_and_reads_like_the_service():
    from biomoqa_rag import pipeline_langgraph as L
    sandbox = object.__new__(L.LangGraphRAGPipeline)   # no graph: langgraph is optional
    sandbox.__dict__.update(pipe().__dict__)
    state = {"question": Q, "collection": "medline", "final_n": 5, "retrieval_n": 20,
             "retrieval": "sparse", "mode": "generative", "debug": False}
    found = sandbox._node_retrieve(state)
    assert [x.doc_id for x in found["documents"]] == ["3", "2", "1"] and found["num_retrieved"] == 3
    read = sandbox._node_generative({**state, "documents": found["documents"]})
    assert read["answers"][0]["answer"] == "Based on the provided documents, Adults are 2 mm long."


def test_multi_collection_ranks_and_reads_each():
    p = pipe()
    r = p.run_multi_collection(Q, mode="generative", depth="fast")
    cols = [c["collection"] for c in r["collection_results"]]
    assert sorted(cols) == ["medline", "plazi", "pmc"]
    assert all(c["depth_used"] == "fast" and c["notes"] == [] for c in r["collection_results"])
    assert r["mode_used"] == "generative" and r["_depth_used"] == ["fast"]
