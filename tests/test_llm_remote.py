"""The client for the shared language-model server, against a local fake server."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from biomoqa_rag.llm_remote import LLMUnavailable, RemoteLLM


class Fake(BaseHTTPRequestHandler):
    behaviour = {"delay": 0.0, "status": 200, "pieces": ["Adults ", "are ", "2 mm [0]."], "error": None}
    last_body = None
    lock = threading.Lock()
    open_now, most_open = 0, 0       # requests being answered at the same time

    def log_message(self, *a):
        pass

    def do_POST(self):
        with Fake.lock:
            Fake.open_now += 1
            Fake.most_open = max(Fake.most_open, Fake.open_now)
        try:
            self._answer()
        finally:
            with Fake.lock:
                Fake.open_now -= 1

    def _answer(self):
        n = int(self.headers.get("Content-Length", 0))
        Fake.last_body = json.loads(self.rfile.read(n))
        b = Fake.behaviour
        if b["status"] != 200:
            self.send_response(b["status"])
            self.end_headers()
            self.wfile.write(b"refused")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # the role chunk comes at once, the first content token after the delay
        self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        self.wfile.flush()
        time.sleep(b["delay"])
        if b["error"]:
            self.wfile.write(b"data: " + json.dumps({"error": b["error"]}).encode() + b"\n\n")
            return
        for p in b["pieces"]:
            chunk = {"choices": [{"delta": {"content": p}, "logprobs": {"content": [{"token": p, "logprob": -0.1}]}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    Fake.behaviour = {"delay": 0.0, "status": 200, "pieces": ["Adults ", "are ", "2 mm [0]."], "error": None}
    Fake.open_now, Fake.most_open = 0, 0
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def call(url, **kw):
    llm = RemoteLLM(url, "Qwen3.8-27B", first_token_timeout=kw.pop("first", 1.0), total_timeout=kw.pop("total", 5.0))
    return llm.chat([{"role": "user", "content": "q"}], max_tokens=50, temperature=0.2, top_p=0.9,
                    prefix="Based on the provided documents, ", logprobs=True, **kw)


def test_streams_the_answer_and_sends_explicit_settings(server):
    c = call(server, stop=["\nQuestion:"], seed=0)
    assert c.text == "Adults are 2 mm [0]." and c.ttft is not None and c.finish_reason == "stop"
    assert c.logprobs == [-0.1, -0.1, -0.1]
    body = Fake.last_body
    assert body["priority"] == -10 and body["repetition_penalty"] == 1.0 and body["presence_penalty"] == 0.0
    assert body["top_k"] == 0 and body["continue_final_message"] is True and body["add_generation_prompt"] is False
    assert body["messages"][-1] == {"role": "assistant", "content": "Based on the provided documents, "}
    assert "min_p" not in body and "logit_bias" not in body


def test_a_call_can_set_its_own_priority(server):
    RemoteLLM(server, "m", priority=-10).chat([{"role": "user", "content": "q"}], max_tokens=5,
                                              temperature=0.0, priority=10)
    assert Fake.last_body["priority"] == 10


def test_no_prefix_turns_thinking_off(server):
    RemoteLLM(server, "m").chat([{"role": "user", "content": "q"}], max_tokens=5, temperature=0.0)
    assert Fake.last_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "continue_final_message" not in Fake.last_body


def test_no_first_token_in_time_is_slow(server):
    Fake.behaviour["delay"] = 1.5
    t0 = time.perf_counter()
    with pytest.raises(LLMUnavailable) as e:
        call(server, first=0.5)
    assert e.value.reason == "slow" and time.perf_counter() - t0 < 1.4


def test_gateway_refusal_is_busy(server):
    Fake.behaviour["status"] = 429
    with pytest.raises(LLMUnavailable) as e:
        call(server)
    assert e.value.reason == "busy"


def test_error_inside_the_stream(server):
    Fake.behaviour["error"] = {"message": "min_p is not supported"}
    with pytest.raises(LLMUnavailable) as e:
        call(server)
    assert e.value.reason == "error"


def test_requests_made_together_reach_the_server_together(server):
    """Nothing in the client makes calls wait for one another, so the server (vLLM) can batch
    every request in flight; with the old in-process model a lock let one through at a time."""
    Fake.behaviour["delay"] = 0.5
    llm = RemoteLLM(server, "Qwen3.8-27B", first_token_timeout=2.0, total_timeout=5.0)
    texts, t0 = [], time.perf_counter()
    threads = [threading.Thread(target=lambda: texts.append(llm.chat(
        [{"role": "user", "content": "q"}], max_tokens=5, temperature=0.0).text)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(texts) == 6 and Fake.most_open == 6
    assert time.perf_counter() - t0 < 1.5      # one answer's time, not six in a row (3 s)


def test_server_down_is_unreachable():
    with pytest.raises(LLMUnavailable) as e:
        call("http://127.0.0.1:9/v1")
    assert e.value.reason in ("unreachable", "slow")
