"""
Client for the shared language-model server (an OpenAI-compatible vLLM endpoint).

The service sends its prompt to the shared GPU server instead of loading a model of its
own, and streams the reply. Two limits keep an interactive answer from hanging when the
server is busy: no first token within `first_token_timeout` seconds, or no complete reply
within `total_timeout`, and the call gives up with LLMUnavailable. The pipeline then
answers with the extractive reader and says so in the response.

Every sampling setting is sent explicitly. The server's own defaults (presence_penalty
1.5, top_k 20, ...) are not what the reader was tested with, and the server rejects
`min_p` and `logit_bias` because it decodes speculatively.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


class LLMUnavailable(Exception):
    """The language model did not answer in time, or could not be reached."""

    def __init__(self, reason: str, detail: str = "", retry_after: Optional[float] = None):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        # "slow": no first token or no complete reply in time, or the gateway timed out (504);
        # "busy": the gateway's caps are full (429) or the model is restarting (503);
        # "unreachable": no connection; "error": any other failed reply.
        self.reason = reason
        # Seconds the gateway asked to wait before trying again (429 and 503 only).
        self.retry_after = retry_after


def _retry_after(resp, default: float = 10.0) -> float:
    """The wait a 429 or 503 asks for (the LLM gateway's 503 sends 10 s)."""
    try:
        return max(0.0, float(resp.headers.get("Retry-After", default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Completion:
    text: str
    ttft: Optional[float]          # seconds to the first content token
    total: float                   # seconds to the end of the reply
    logprobs: List[float] = field(default_factory=list)
    finish_reason: Optional[str] = None


class RemoteLLM:
    """Chat completions against the shared server, one streamed request per call."""

    def __init__(
        self,
        base_url: str,
        model: str,
        first_token_timeout: float = 4.0,
        total_timeout: float = 20.0,
        priority: int = -10,
        connect_timeout: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.first_token_timeout = first_token_timeout
        self.total_timeout = total_timeout
        self.priority = priority
        self.connect_timeout = connect_timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def chat(
        self,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        prefix: Optional[str] = None,
        response_format: Optional[dict] = None,
        logprobs: bool = False,
        first_token_timeout: Optional[float] = None,
        total_timeout: Optional[float] = None,
        priority: Optional[int] = None,
        retry_busy: bool = False,
    ) -> Completion:
        """One completion. `prefix` pre-fills the assistant turn; the reply continues it
        and does not repeat it. `priority` overrides the client's (batch jobs send a
        positive one, so the server serves people first). With `retry_busy`, a 429 or
        503 is waited out as the gateway asks (Retry-After) and the request sent again,
        while the whole call stays within `total_timeout`. Raises LLMUnavailable on any
        failure or time limit."""
        first_limit = self.first_token_timeout if first_token_timeout is None else first_token_timeout
        total_limit = self.total_timeout if total_timeout is None else total_timeout
        msgs = list(messages)
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "priority": self.priority if priority is None else priority,
            "stream": True,
        }
        if prefix:
            # Continuing an English sentence also keeps the model from thinking first.
            msgs.append({"role": "assistant", "content": prefix})
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body["messages"] = msgs
        if stop:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed
        if response_format:
            body["response_format"] = response_format
        if logprobs:
            body["logprobs"] = True

        start = time.perf_counter()
        while True:
            left = total_limit - (time.perf_counter() - start)
            try:
                return self._attempt(body, min(first_limit, left), left)
            except LLMUnavailable as e:
                left = total_limit - (time.perf_counter() - start)
                # The gateway says when to come back (429: its request caps are full; 503: the
                # model is restarting). A batch job waits and asks again, as the server's
                # documentation expects of clients, as long as the wait fits in its time
                # limit; a person's request does not wait (its answer is sent at once).
                if not (retry_busy and e.retry_after is not None and e.retry_after + 1.0 < left):
                    raise
                time.sleep(e.retry_after)

    def _attempt(self, body: dict, first_limit: float, total_limit: float) -> Completion:
        """One request to the server, streamed; LLMUnavailable on any failure or time limit."""
        t0 = time.perf_counter()
        # The read timeout bounds every wait for the next bytes: before the first token
        # it is the first-token limit, and a stall in the middle of a reply ends it too.
        try:
            resp = self._session().post(
                f"{self.base_url}/chat/completions", json=body, stream=True,
                timeout=(self.connect_timeout, first_limit),
            )
        except requests.exceptions.ConnectionError as e:
            if isinstance(e, requests.exceptions.ConnectTimeout) or "timed out" in str(e).lower():
                raise LLMUnavailable("slow", "no connection in time") from e
            raise LLMUnavailable("unreachable", str(e)[:200]) from e
        except requests.exceptions.Timeout as e:
            raise LLMUnavailable("slow", "no reply in time") from e
        except requests.exceptions.RequestException as e:
            raise LLMUnavailable("unreachable", str(e)[:200]) from e

        with resp:
            if resp.status_code in (429, 503):
                why = "its request caps are full" if resp.status_code == 429 else "the model is restarting"
                raise LLMUnavailable("busy", f"the server refused the request, {why} ({resp.status_code})",
                                     retry_after=_retry_after(resp))
            if resp.status_code == 504:
                # The gateway stopped waiting for the model; it marks this reply not to be retried.
                raise LLMUnavailable("slow", "the gateway timed out (504)")
            if resp.status_code != 200:
                raise LLMUnavailable("error", f"HTTP {resp.status_code}: {resp.text[:200]}")
            parts: List[str] = []
            lps: List[float] = []
            ttft: Optional[float] = None
            finish: Optional[str] = None
            try:
                for line in resp.iter_lines():
                    now = time.perf_counter() - t0
                    if ttft is None and now > first_limit:
                        raise LLMUnavailable("slow", f"no first token in {first_limit:g} s")
                    if now > total_limit:
                        raise LLMUnavailable("slow", f"no complete reply in {total_limit:g} s")
                    if not line or not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    chunk = json.loads(payload)
                    if chunk.get("error"):
                        # vLLM reports a rejected streaming request inside a 200 stream.
                        raise LLMUnavailable("error", json.dumps(chunk["error"])[:200])
                    for choice in chunk.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            parts.append(piece)
                        for entry in ((choice.get("logprobs") or {}).get("content") or []):
                            lp = entry.get("logprob")
                            if isinstance(lp, (int, float)):
                                lps.append(float(lp))
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
            except LLMUnavailable:
                raise
            except requests.exceptions.RequestException as e:
                kind = "slow" if "timed out" in str(e).lower() or isinstance(e, requests.exceptions.Timeout) else "error"
                raise LLMUnavailable(kind, str(e)[:200]) from e
            except ValueError as e:  # a line that is not JSON
                raise LLMUnavailable("error", f"unreadable reply: {e}") from e

        return Completion(text="".join(parts), ttft=ttft, total=time.perf_counter() - t0,
                          logprobs=lps, finish_reason=finish)
