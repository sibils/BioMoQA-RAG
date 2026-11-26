"""
FastAPI server for BioMoQA RAG pipeline.

Configuration is read from config.toml in the working directory.
"""

# Must be set before any CUDA/vLLM initialization.
# VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps EngineCore in the same process.
# The V1 engine (default in vLLM >=0.7) otherwise spawns a subprocess for CUDA
# which fails on this VM's CUDA driver ("operation not supported").
# VLLM_USE_V1 was removed in vLLM >=0.9 — use VLLM_ENABLE_V1_MULTIPROCESSING instead.
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# Use a fixed shared HF cache so ansible pre-download and the service use the same path.
# HF_HOME must be set before any huggingface_hub / transformers imports.
os.environ.setdefault("HF_HOME", "/opt/sibils-qa/.cache/huggingface")
# Use local HuggingFace cache — avoids 429 rate-limit errors on startup
# when vLLM checks the repo file list. Model must be pre-downloaded.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# MIG / vGPU workaround for sibils-prod-ai (GRID A100D-1-20C in MIG mode):
#
# 1. Auto-detect the MIG partition UUID via nvidia-smi so the service file does not
#    need to hardcode it (MIG UUIDs change on VM reboot).
#    CUDA_VISIBLE_DEVICES must be set to the UUID so cuda:0 maps to the MIG partition.
#    But vLLM 0.17 tries to int() that string during config creation → pydantic error.
#    Fix: warm-up CUDA with the UUID first, then reset to "0" (vLLM-parseable).
#    CUDA keeps the context bound to the MIG device for the process lifetime.
#
# 2. PyTorch 2.4+ enables expandable_segments (VMM via cuMemCreate) by default.
#    VMM is NOT supported on NVIDIA vGPU/GRID → "CUDA driver error: operation not supported".
#    Fix: disable expandable_segments before the first CUDA call.
def _setup_mig_cuda():
    import subprocess, re
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        m = re.search(r"UUID: (MIG-[0-9a-f-]+)\)", out)
        if not m:
            return  # No MIG device found, nothing to do
        mig_uuid = m.group(1)
    except Exception:
        return  # nvidia-smi unavailable or failed

    os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuid
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    try:
        import torch
        torch.zeros(1, device="cuda:0")  # init CUDA context with MIG partition
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # reset to numeric for vLLM config
    except Exception:
        del os.environ["CUDA_VISIBLE_DEVICES"]  # revert on failure

_setup_mig_cuda()

import dataclasses
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict

import requests
import threading
import time as _time
from .config import get_config
from .depth import DepthSettings
from .pipeline import RAGPipeline, RAGConfig


# Per-question cache for multi-collection results.
# The frontend calls /qa 4× in parallel (one per collection tab).
# The first call runs the full pipeline and caches; the other 3 wait and
# return from cache immediately, avoiding 4× sequential vLLM generations.
_multi_cache: dict = {}           # (question, mode, depth, sizes, batch) -> (result, timestamp, ttl)
_multi_in_flight: dict = {}       # same key -> threading.Event
_multi_cache_mutex = threading.Lock()
_MULTI_CACHE_TTL = 120            # seconds
# An answer given with a fallback note says "try again in a moment": kept only long enough for
# the calls already waiting on it, so a retry gets a fresh run.
_DEGRADED_TTL = 10
_DEGRADED_CODES = {"extractive_fallback", "depth_reduced"}


def _is_batch(http_request: Request) -> bool:
    """A batch job is a POST that does not come from a web browser.

    Every current browser sends Sec-Fetch-Mode with each request, and Origin with a POST;
    scripts such as the Kybele trait pipeline send neither. A batch job's answer is written at
    low priority on the shared server, may wait ([llm] batch_* in config.toml), and fails with
    `error` rather than falling back to BioBERT. GET /QA, the web page's endpoint, never is one.
    """
    h = http_request.headers
    return http_request.method == "POST" and not (h.get("sec-fetch-mode") or h.get("origin"))


def _get_or_run_multi(pipeline, question, mode, retrieval, retrieval_n, final_n, debug, depth=None,
                      batch=False):
    key = (question, mode, depth, retrieval_n, final_n, batch)
    with _multi_cache_mutex:
        if key in _multi_cache:
            result, ts, ttl = _multi_cache[key]
            if _time.time() - ts < ttl:
                return result
        if key in _multi_in_flight:
            event = _multi_in_flight[key]
            is_runner = False
        else:
            event = threading.Event()
            _multi_in_flight[key] = event
            is_runner = True
    if is_runner:
        try:
            result = pipeline.run_multi_collection(
                question=question, retrieval_n=retrieval_n,
                final_n=final_n, mode=mode, retrieval=retrieval, debug=debug, depth=depth,
                batch=batch,
            )
            degraded = bool(_DEGRADED_CODES & set(result.get("_notes") or []))
            with _multi_cache_mutex:
                _multi_cache[key] = (result, _time.time(), _DEGRADED_TTL if degraded else _MULTI_CACHE_TTL)
                _multi_in_flight.pop(key, None)
            event.set()
            return result
        except Exception:
            with _multi_cache_mutex:
                _multi_in_flight.pop(key, None)
            event.set()
            raise
    else:
        event.wait(timeout=180)
        with _multi_cache_mutex:
            if key in _multi_cache:
                return _multi_cache[key][0]
        raise RuntimeError("Multi-collection computation did not complete")

# Load configuration
config = get_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.server.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("biomoqa")


class _NoQueryString(logging.Filter):
    """Keep questions out of the access log: GET /QA carries the question in its query
    string, and the service logs times and codes, never what people asked."""

    def filter(self, record):
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            record.args = args[:2] + (args[2].split("?", 1)[0],) + args[3:]
        return True


logging.getLogger("uvicorn.access").addFilter(_NoQueryString())


def _log_request(endpoint: str, **fields) -> None:
    """One line per question: depth, mode, collection, timings and fallback codes.
    Never the question text."""
    logger.info("qa_request %s", json.dumps({"endpoint": endpoint, **fields}, default=str))


def _log_doc_refs(endpoint: str, mode: str, batch: bool, result: dict) -> None:
    _log_request(endpoint + " doc_refs", mode=mode, batch=batch, mode_used=result.get("mode_used"),
                 notes=sorted({n["code"] for r in result.get("collection_results", []) for n in r.get("notes") or []}),
                 n_docs=result.get("ndocs_retrieved"), seconds=result.get("pipeline_time"), success=True)


def _running_commit() -> Optional[str]:
    """The commit this code was installed from: pip records it for an install from git
    (direct_url.json); in a working tree, git's HEAD. None when neither is known."""
    try:
        from importlib.metadata import distribution
        raw = distribution("biomoqa-rag").read_text("direct_url.json")
        if raw:
            commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
            if commit:
                return commit
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


_COMMIT = _running_commit()

# ---------------------------------------------------------------------------
# OpenAPI metadata
# ---------------------------------------------------------------------------

_APP_DESCRIPTION = """
## BioMoQA RAG — Biomedical Question Answering

Answers natural-language biomedical questions from three literature collections. For each
collection the service searches at the **depth** asked, keeps the five documents that best
match the question, and reads them with a **language model** (generative) or **BioBERT**
(extractive).

---

### Pipeline

```
Question
  └─▶ Searches per collection
        fast: the query parser's query + a plain OR search of the question's words
        deep: the same + two searches written by the language model
               (+ one on its guess, when it is sure of it)
  └─▶ Cross-encoder reranker: every candidate scored against the question as asked
  └─▶ The five best documents
  └─▶ Generative (language model on the shared GPU server) or extractive (BioBERT)
  └─▶ Answer + the documents it cites
```

When the language model is slow or unavailable, a generative request gets the extractive
answer at once, with `mode_used: "extractive"` and a note in `notes` saying so.

---

### Collections

| ID | Source | Content |
|----|--------|---------|
| `medline` | PubMed / MEDLINE | Biomedical abstracts |
| `pmc` | PubMed Central | Full-text open-access articles |
| `plazi` | Plazi | Biodiversity & taxonomy literature |

`GET /QA` answers for one collection (`col`); `POST /qa` for all three, ranked. Use
`doc_refs` to bypass retrieval entirely and target specific documents.

---

### Depth (`depth`)

| `depth` | What it does | Typical time: one collection · three at once |
|---------|--------------|--------------|
| `fast` *(default)* | Two keyword searches merged, reranked, five documents read | about 3 s · 6 s |
| `deep` | Also searches written by the language model; the reranker still keeps five | about 5 s · 8 s |

Each response says what ran (`depth_used`) and, in `notes`, when the service did less than
asked. `retrieval` (`sparse`, `dense`) is still accepted and ignored.

---

### Answer modes (`mode`)

| `mode` | Reader | Output |
|--------|--------|--------|
| `generative` | Language model (shared GPU server) | A short answer; `docs` lists the documents it cites |
| `extractive` | BioBERT | A span copied from a document, with offsets and a confidence score |

---

### Direct-document QA (`doc_refs`)

Pass a list of PMIDs, PMCIDs, DOIs, or title fragments in `doc_refs` to skip retrieval
and run QA directly on those documents. Useful when you already know which papers are relevant.
"""

_TAGS_METADATA = [
    {
        "name": "Question Answering",
        "description": (
            "Submit a biomedical question and receive answers backed by document citations. "
            "`POST /qa` (recommended) queries all three collections in one request and returns "
            "results ranked by relevance. `GET /QA` is a legacy single-collection endpoint "
            "backwards-compatible with the old SIBILS QA API."
        ),
    },
    {
        "name": "Batch",
        "description": (
            "Submit multiple questions in a single request. Retrieval runs in parallel across "
            "all questions; generation is sequential. Useful for bulk evaluation or dataset annotation."
        ),
    },
    {
        "name": "System",
        "description": "Health check and pipeline introspection endpoints.",
    },
]

# Initialize app
app = FastAPI(
    title="BioMoQA RAG API",
    description=_APP_DESCRIPTION,
    version="1.0.0",
    root_path="/api",
    docs_url="/",
    openapi_tags=_TAGS_METADATA,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (lazy loading)
pipeline = None


def get_pipeline():
    """Lazy load pipeline based on config.toml configuration"""
    global pipeline
    if pipeline is None:
        logger.info("Initializing BioMoQA RAG pipeline...")
        common = dict(
            sibils_collections=config.sibils.collections,
            sibils_cache_dir=config.sibils.cache_dir,
            sibils_cache_ttl=config.sibils.cache_ttl,
            sibils_empty_cache_ttl=config.sibils.empty_cache_ttl,
            sibils_enforce_taxon_phrase=config.sibils.enforce_taxon_phrase,
            default_depth=config.depth.default,
            depth=DepthSettings(
                candidates_per_search=config.depth.candidates_per_search,
                subquery_timeout=config.depth.subquery_timeout,
                extra_search_timeout=config.depth.extra_search_timeout,
                min_agreement=config.depth.min_agreement,
                max_new_words=config.depth.max_new_words,
                guess_query=config.depth.guess_query,
                guess_margin=config.depth.guess_margin,
                batch_subquery_timeout=config.depth.batch_subquery_timeout,
            ),
            reranker_model=config.reranking.model,
            reranker_device=config.reranking.device,
            final_n=config.relevance_filter.final_n,
            min_extractive_score=config.relevance_filter.min_extractive_score,
            temperature=config.generation.temperature,
            max_abstract_length=config.context.max_abstract_length,
            llm_abstract_length=config.context.llm_abstract_length,
            llm_whole_upto=config.context.llm_whole_upto,
            truncate_abstracts=config.context.truncate_abstracts,
            qa_device=config.extraction.device,
            qa_model=config.extraction.model,
            llm_base_url=config.llm.base_url,
            llm_model=config.llm.model,
            llm_first_token_timeout=config.llm.first_token_timeout,
            llm_total_timeout=config.llm.total_timeout,
            llm_priority=config.llm.priority,
            llm_batch_priority=config.llm.batch_priority,
            llm_batch_first_token_timeout=config.llm.batch_first_token_timeout,
            llm_batch_total_timeout=config.llm.batch_total_timeout,
        )
        backend = config.generation.backend
        if backend == "remote":
            logger.info(f"Generation: shared server {config.llm.base_url} ({config.llm.model})")
            rag_config = RAGConfig(generation_backend="remote", max_tokens=config.generation.max_tokens, **common)
        elif backend == "cpu" or config.model.mode == "cpu":
            logger.info(f"Generation: CPU inference, model size: {config.model.size}")
            rag_config = dataclasses.replace(RAGConfig.cpu_config(model_size=config.model.size), **common)
        elif config.model.mode == "gpu_small":
            logger.info("Generation: GPU (small model, ~8GB VRAM)")
            rag_config = dataclasses.replace(RAGConfig.gpu_small_config(), **common)
        else:
            logger.info(f"Generation: GPU in this process, model: {config.model.model_name}")
            rag_config = RAGConfig(
                generation_backend="vllm",
                model_name=config.model.model_name,
                quantization=config.model.quantization,
                gpu_memory_utilization=config.model.gpu_memory_utilization,
                max_model_len=config.model.max_model_len,
                max_tokens=config.generation.max_tokens,
                **common,
            )

        pipeline = RAGPipeline(rag_config)
        logger.info("Pipeline ready")
    return pipeline


_doc_resolver = None  # initialised lazily on first use, after pipeline is ready


def get_doc_resolver():
    global _doc_resolver
    if _doc_resolver is None:
        from .retrieval.doc_resolver import DocResolver
        _doc_resolver = DocResolver(get_pipeline().sibils_retriever)
    return _doc_resolver


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """Request body for `POST /qa` and `POST /qa/multi`."""

    question: str = Field(
        ...,
        description="Natural-language biomedical question.",
        examples=["What causes malaria?", "How does CRISPR-Cas9 edit DNA?"],
    )
    retrieval_n: Optional[int] = Field(
        default=None,
        ge=1, le=100,
        description=(
            "Documents fetched by each search before reranking (default: 20, max: 100). "
            "More candidates cost reranking time (about 0.8 s per 20 documents on CPU)."
        ),
    )
    final_n: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of documents the reader gets after reranking (default: 5). "
            "On the test questions, 3 to 10 documents gave the same answers within chance."
        ),
    )
    depth: Optional[Literal["fast", "deep"]] = Field(
        default=None,
        description=(
            "How hard to search:\n"
            "- `fast` *(default)* — two keyword searches per collection, merged and reranked; "
            "about 3 s for one collection, 6 s for three\n"
            "- `deep` — also searches written by the language model; about 5 s for one collection, 8 s for three\n"
            "`depth_used` in each result says what actually ran."
        ),
    )
    include_documents: bool = Field(
        default=False,
        description=(
            "When `true`, the full abstract or document text is included in `doc_text` "
            "for every `DocItem` in the response. Increases response size significantly."
        ),
    )
    debug: bool = Field(
        default=False,
        description=(
            "When `true`, adds internal pipeline diagnostics to `debug_info` in each "
            "`CollectionResult`: per-document scores, reranker output, timing breakdown, etc."
        ),
    )
    mode: str = Field(
        default="generative",
        description=(
            "Answer mode:\n"
            "- `generative` *(default)* — the language model writes a short answer; `docs` lists "
            "the documents it cites. When the model is slow or down, the extractive answer is sent "
            "instead and `notes` says so\n"
            "- `extractive` — BioBERT copies a span from the best matching document "
            "and returns a confidence score\n"
            "- `hybrid` *(legacy alias)* — same as `generative`"
        ),
    )
    retrieval: Optional[str] = Field(
        default=None,
        description="Ignored. Kept so that older clients sending `sparse` or `dense` still work; use `depth`.",
    )
    doc_refs: Optional[List[str]] = Field(
        default=None,
        description=(
            "Skip retrieval and run QA on specific documents. Each entry can be a PMID "
            "(e.g. `12345678`), PMCID (e.g. `PMC9712345`), DOI, or title fragment. "
            "Documents are fetched from SIBILS, grouped by collection, and QA is run per group."
        ),
    )

    model_config = {"json_schema_extra": {"example": {
        "question": "What causes malaria?",
        "mode": "generative",
        "depth": "fast",
    }}}


class DocItem(BaseModel):
    """A source document attached to an answer."""

    docid: Optional[str] = Field(
        None,
        description="Document identifier: PMID for MEDLINE, PMCID for PMC, or Plazi UUID.",
    )
    doc_source: Optional[str] = Field(
        None,
        description="Collection the document comes from: `medline`, `pmc`, or `plazi`.",
    )
    doc_retrieval_score: Optional[float] = Field(
        None,
        description=(
            "The reranker's probability that the document answers the question, 0 to 1 "
            "(higher = more relevant). Comparable across collections."
        ),
    )
    doc_text: Optional[str] = Field(
        None,
        description="Full abstract or document text. Only populated when `include_documents=true`.",
    )
    snippet_start: Optional[int] = Field(
        None,
        description="Character offset of the answer span start within `doc_text`. Extractive mode only.",
    )
    snippet_end: Optional[int] = Field(
        None,
        description="Character offset of the answer span end within `doc_text`. Extractive mode only.",
    )
    passage: Optional[str] = Field(
        None,
        description=(
            "The passage of this document the reader was given: what the answer can rest on. "
            "For full-text documents it is a window chosen around the question's terms, not "
            "the start of the document."
        ),
    )


class AnswerItem(BaseModel):
    """A single answer candidate with its supporting documents."""

    answer: str = Field(
        ...,
        description=(
            "The answer text. "
            "In **generative** mode: two to four sentences written by the language model; the "
            "citation markers are removed and the documents it cites are in `docs`. "
            "In **extractive** mode: a verbatim span copied from the source document."
        ),
    )
    answer_score: Optional[float] = Field(
        None,
        description=(
            "BioBERT span confidence score between 0 and 1. "
            "Only present in extractive mode; `null` for generative answers."
        ),
    )
    generation_confidence: Optional[float] = Field(
        None,
        description=(
            "Generative mode: geometric mean of the generated tokens' probabilities, between "
            "0 and 1. A rough signal, not calibrated, and not comparable with `answer_score`."
        ),
    )
    hedged: Optional[bool] = Field(
        None,
        description=(
            "Generative mode: the answer commits only partly (\"may\", \"possibly\", "
            "\"suggests\"). Hedged answers are wrong more often than plain ones."
        ),
    )
    states_no_answer: Optional[bool] = Field(
        None,
        description="Generative mode: the answer says the sources do not contain what was asked.",
    )
    docs: List[DocItem] = Field(
        default_factory=list,
        description=(
            "Supporting documents for this answer. "
            "Extractive: exactly one item with `snippet_start`/`snippet_end` offsets. "
            "Generative: the documents the answer cites, in rank order (the top document "
            "when it cites none)."
        ),
    )


class Note(BaseModel):
    """Something the person reading the answer should know."""

    code: str = Field(
        ...,
        description=(
            "`depth_reduced`: the deep mode's extra searches were skipped (the language model "
            "was too slow), so the answer comes from the fast search. `extractive_fallback`: "
            "the language model did not answer in time, so the answer is the extractive one. "
            "`deep_unavailable`: this server has no deep mode. More codes may come."
        ),
    )
    message: str = Field(..., description="A sentence to show as it is.")


_DEPTH_USED = Field(
    None,
    description=(
        "The depth that actually ran: `fast` or `deep`. Lower than asked when the deep "
        "mode's extra searches were skipped; `notes` then says why. Null with `doc_refs`."
    ),
)
_NOTES = Field(
    default_factory=list,
    description="Notes for the person reading the answer, mostly when the service did less than asked. Often empty.",
)
_QUERIES_USED = Field(
    default_factory=list,
    description="Deep only: the searches the language model wrote and the service kept. Empty for fast.",
)


class CollectionResult(BaseModel):
    """QA result for one literature collection."""

    collection: str = Field(
        ...,
        description="Collection identifier: `medline`, `pmc`, or `plazi`.",
    )
    rank: int = Field(
        ...,
        description=(
            "Rank of this collection for the question (1 = best). A collection whose answer "
            "answers ranks above one whose answer says the sources do not state it; for a "
            "question naming a species, Plazi ranks first among those; then plain answers "
            "rank above hedged ones. Remaining ties keep retrieval order (with `doc_refs`: "
            "the answer's score)."
        ),
    )
    answers: List[AnswerItem] = Field(
        ...,
        description="Answer candidates for this collection, ordered best-first.",
    )
    mode_used: Optional[str] = Field(
        None,
        description=(
            "The mode that wrote these answers: `generative` or `extractive`. Extractive "
            "after a generative request when the language model was slow (see `notes`)."
        ),
    )
    depth_used: Optional[str] = _DEPTH_USED
    notes: List[Note] = _NOTES
    queries_used: List[str] = _QUERIES_USED
    debug_info: Optional[Dict] = Field(
        None,
        description=(
            "Internal pipeline diagnostics: per-document scores, reranker output, timing. "
            "Only present when `debug=true`."
        ),
    )


class MultiQAResponse(BaseModel):
    """Response from `POST /qa` and `POST /qa/multi` — results from all collections ranked best-first."""

    question: str = Field(..., description="The question as submitted.")
    collection_results: List[CollectionResult] = Field(
        ...,
        description=(
            "Per-collection QA results, sorted by relevance rank (rank 1 first). "
            "A collection is omitted if no documents were retrieved from it."
        ),
    )
    mode_used: str = Field(
        ...,
        description="Effective answer mode used for this request (`generative` or `extractive`).",
    )
    ndocs_retrieved: int = Field(
        ...,
        description="Total number of documents retrieved across all collections before relevance filtering.",
    )
    model: str = Field(
        ...,
        description=(
            "Short identifier of the model that produced the answers: "
            "`biobert` for extractive mode, or the generative model name (e.g. `qwen3.8-27b`)."
        ),
    )
    pipeline_time: Optional[float] = Field(
        None,
        description="Total wall-clock time for the full pipeline in seconds.",
    )
    error: Optional[str] = Field(
        None,
        description=(
            "Set when the pipeline failed. An empty `collection_results` with no `error` "
            "means the question genuinely retrieved nothing; with an `error` it means the "
            "request failed and the result is not an answer."
        ),
    )
    unresolved_refs: Optional[List[str]] = Field(
        None,
        description=(
            "Entries from `doc_refs` that could not be resolved to a document in SIBILS. "
            "`null` when all refs were found or when `doc_refs` was not used."
        ),
    )


class QAResponse(BaseModel):
    """
    Response from `GET /QA` — single-collection format, backwards-compatible with
    `biodiversitypmc.sibils.org/api/QA`.
    """

    sibils_version: str = Field(..., description="API version string, e.g. `biomoqa-2.0`.")
    success: bool = Field(..., description="`true` if the pipeline completed without error.")
    error: str = Field(..., description="Error description when `success=false`; empty string on success.")
    question: str = Field(..., description="The question as submitted.")
    collection: str = Field(..., description="Collection queried.")
    model: str = Field(
        ...,
        description="Model used: `biobert` for extractive mode, or short generative model name.",
    )
    ndocs_requested: int = Field(..., description="Number of documents the reader was to get (`n`, at most 5).")
    ndocs_returned_by_SIBiLS: int = Field(..., description="Number of different documents the searches returned, before reranking.")
    answers: List[AnswerItem] = Field(..., description="Ranked answer candidates, best first.")
    mode_used: Optional[str] = Field(
        None,
        description=(
            "The mode that wrote the answers: `generative` or `extractive`. Extractive after a "
            "generative request when the language model was slow (see `notes`)."
        ),
    )
    depth_used: Optional[str] = _DEPTH_USED
    notes: List[Note] = _NOTES
    queries_used: List[str] = _QUERIES_USED
    pipeline_time: Optional[float] = Field(None, description="Total pipeline wall-clock time in seconds.")
    transformed_query: None = Field(None, description="Not used (reserved for Elasticsearch-based backends).")
    debug_info: Optional[Dict] = Field(None, description="Internal diagnostics when `debug=true`.")
    documents: Optional[List[Dict]] = Field(None, description="Full document objects when `include_documents=true`.")


class BatchRequest(BaseModel):
    """Request body for `POST /batch`."""

    questions: List[str] = Field(
        ...,
        min_length=1,
        description="List of questions to answer. Each is processed independently with parallel retrieval.",
    )
    retrieval_n: Optional[int] = Field(
        None,
        ge=1, le=100,
        description="Documents to retrieve per question (default: 10, max: 100).",
    )
    final_n: Optional[int] = Field(
        None,
        ge=1,
        description="Documents kept after relevance filtering per question (default: 5).",
    )
    col: Optional[str] = Field(
        None,
        description='Restrict to a single collection: `"medline"`, `"pmc"`, or `"plazi"`. Defaults to all.',
    )

    model_config = {"json_schema_extra": {"example": {
        "questions": [
            "What causes malaria?",
            "What diseases are associated with ticks?",
        ],
        "retrieval_n": 10,
        "col": "medline",
    }}}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Initialize pipeline on startup"""
    logger.info("=" * 60)
    logger.info("Starting BioMoQA RAG API")
    logger.info("=" * 60)
    p = get_pipeline()
    _ = p.extractor  # warm up BioBERT so first request is not slow


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Service status and loaded model information",
)
def health_check():
    """
    Returns the service status, the commit the code was installed from, and, once the
    pipeline is initialised, the models it uses.

    The `ready` field is `false` during startup (pipeline not yet loaded) and `true`
    once the pipeline is loaded.
    """
    config_info = {}
    if pipeline is not None:
        config_info = {
            "generation_backend": pipeline.config.generation_backend,
            "generative_model": pipeline.config.generator_label,
            "extractive_model": pipeline.config.qa_model,
            "reranker_model": pipeline.config.reranker_model,
            "default_depth": pipeline.config.default_depth,
        }

    return {
        "status": "healthy",
        "commit": _COMMIT,
        "features": {
            "depths": ["fast", "deep"],
            "cross_encoder_reranking": True,
            "extractive_fallback": True,
            "sentence_citations": True,
        },
        "ready": pipeline is not None,
        "config": config_info
    }


@app.get(
    "/retrieval-info",
    tags=["System"],
    summary="Retrieval system overview",
    response_description="Description of the hybrid retrieval pipeline and its components",
)
def retrieval_info():
    """
    Describes the two search depths: what each one runs, and what the tests behind them
    measured.
    """
    return {
        "depths": {
            "fast": {
                "default": True,
                "searches": [
                    "the SIBILS query parser's query for the collection (concept annotations, species filter)",
                    "a plain OR search of the question's words (species filter kept)",
                ],
                "candidates_per_search": 20,
                "ranking": "a cross-encoder scores every candidate against the question as asked; the best five are read",
                "typical_time": "about 3 s for one collection, 6 s for three at once",
            },
            "deep": {
                "searches": [
                    "the two searches of fast",
                    "two searches written by the language model from the question",
                    "one search on the model's own guess, only when it is sure of it",
                ],
                "rules": [
                    "a model-written search is used only if its first five documents match the question about as well as the plain searches do",
                    "a document only the guess search found must clearly beat the fifth document",
                    "the reader never sees the guess or the queries",
                    "no reply from the model within 3 s: the fast search alone is used, and a note says so",
                ],
                "typical_time": "about 5 s for one collection, 8 s for three at once",
            },
        },
        "measured": (
            "258 test questions on Medline, run through this service on 26 November 2026: the right "
            "document was among the five for 75 of 256 questions with fast against 41 with the former "
            "single search, and the answer text for 203 of 258 against 147. The generated answer held "
            "the expected one for 164 questions with fast and 168 with deep (8 gained, 4 lost: could "
            "be chance). Medians with nothing cached: fast 3.1 s for one collection and 5.6 s for "
            "three at once, deep 5.1 s and 8.4 s; the reranker, on the CPU, is the slowest step."
        ),
        "retrieval_parameter": "`retrieval` (sparse, dense) is accepted and ignored; use `depth`.",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_RETRIEVAL_N = 100


def _run_doc_refs_qa(
    question: str,
    doc_refs: List[str],
    mode: str,
    debug: bool = False,
    include_documents: bool = False,
    batch: bool = False,
) -> dict:
    """Resolve doc_refs to Documents, run QA, return a MultiQAResponse-shaped dict."""
    import time as _t

    p = get_pipeline()
    resolver = get_doc_resolver()
    t0 = _t.time()

    docs, unresolved = resolver.resolve_batch(doc_refs)

    if not docs:
        return {
            "question": question,
            "collection_results": [],
            "mode_used": mode,
            "ndocs_retrieved": 0,
            "model": "",
            "pipeline_time": round(_t.time() - t0, 3),
            "unresolved_refs": unresolved if unresolved else None,
        }

    # Group docs by collection and run QA per group (same shape as run_multi_collection)
    from collections import defaultdict
    by_col: dict = defaultdict(list)
    for doc in docs:
        by_col[doc.source].append(doc)

    def read_one(group):
        col, col_docs = group
        result = p.run_with_documents(
            question=question,
            documents=col_docs,
            mode=mode,
            return_documents=include_documents,
            debug=debug,
            batch=batch,
        )
        return {
            "collection": col,
            "answers": result.get("answers", []),
            "mode_used": result.get("mode_used"),
            "depth_used": None,
            "notes": result.get("notes", []),
            "debug_info": result.get("debug_info") if debug else None,
        }

    # The collections are read at the same time, as in run_multi_collection: the shared
    # language-model server batches requests that arrive together, so three collections
    # cost about one answer's time instead of three in a row.
    from concurrent.futures import ThreadPoolExecutor
    groups = sorted(by_col.items())
    with ThreadPoolExecutor(max_workers=min(3, len(groups))) as pool:
        collection_results = list(pool.map(read_one, groups))
    # Rank by the answers, not alphabetically: with sorted() Medline came first whenever
    # present, and its "not stated" displaced the value a Plazi treatment gave (D10.3 P19).
    from .answer_signals import rank_collection_results
    collection_results = rank_collection_results(question, collection_results, use_scores=True)

    # What answered, not what was asked: the language model the configuration uses (the
    # shared server's, not a local model name), or BioBERT when the reply fell back to it.
    mode_used = collection_results[0]["mode_used"] or mode
    return {
        "question": question,
        "collection_results": collection_results,
        "mode_used": mode_used,
        "ndocs_retrieved": len(docs),
        "model": RAGPipeline._model_label(mode_used, p.config.generator_label),
        "pipeline_time": round(_t.time() - t0, 3),
        "unresolved_refs": unresolved if unresolved else None,
    }


def _public(result: dict) -> dict:
    """The response without the pipeline's internal keys (timings for the log)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _clamp_retrieval_n(n: Optional[int]) -> Optional[int]:
    if n is None:
        return None
    return max(1, min(n, _MAX_RETRIEVAL_N))


def _resolve_mode(mode: str, retrieval: Optional[str]) -> tuple[str, Optional[str]]:
    """Normalise the mode; `retrieval` is passed through and ignored by the pipeline."""
    if mode == "hybrid":  # legacy alias
        return "generative", retrieval
    return mode, retrieval


def _run_qa(
    question: str,
    col: Optional[str],
    n: Optional[int],
    mode: str = "generative",
    include_documents: bool = False,
    debug: bool = False,
    retrieval_n: Optional[int] = None,
    final_n: Optional[int] = None,
    depth: Optional[str] = None,
) -> QAResponse:
    """Shared logic for GET /QA: one collection (or all configured ones, merged).

    `n` is how many documents the reader gets, at most the configured final_n (5).
    """
    from .retrieval.sibils_retriever import SIBILSRetriever
    if col is not None and col not in SIBILSRetriever.VALID_COLLECTIONS:
        return QAResponse(
            sibils_version="biomoqa-2.0", success=False,
            error=f"Invalid collection '{col}'. Valid: {', '.join(sorted(SIBILSRetriever.VALID_COLLECTIONS))}",
            question=question, collection=col or "", model="", ndocs_requested=0,
            ndocs_returned_by_SIBiLS=0, answers=[],
        )

    try:
        p = get_pipeline()
        if final_n is None and n:
            final_n = max(1, min(n, p.config.final_n))
        result = p.run(
            question=question,
            retrieval_n=retrieval_n,
            final_n=final_n,
            collection=col,
            return_documents=include_documents,
            debug=debug,
            mode=mode,
            depth=depth,
        )
        _log_request("GET /QA", collection=col, mode=mode, mode_used=result.get("mode_used"),
                     depth=depth or p.config.default_depth, depth_used=result.get("depth_used"),
                     notes=[nt["code"] for nt in result.get("notes", [])],
                     n_answers=len(result.get("answers", [])), seconds=result.get("pipeline_time"),
                     timings=result.get("_timings"), success=True)
        return QAResponse(**_public(result))
    except Exception as e:
        logger.exception("Pipeline error")
        _log_request("GET /QA", collection=col, mode=mode, depth=depth, success=False,
                     error=type(e).__name__)
        return QAResponse(
            sibils_version="biomoqa-2.0", success=False, error=str(e),
            question=question, collection=col or "medline+plazi", model="",
            ndocs_requested=0, ndocs_returned_by_SIBiLS=0, answers=[],
        )


# ---------------------------------------------------------------------------
# Question Answering endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/qa",
    tags=["Question Answering"],
    summary="Answer a question across all collections (recommended)",
    response_model=MultiQAResponse,
    response_description="Answers from all three collections, ranked by relevance",
)
def answer_question_post(request: QuestionRequest, http_request: Request):
    """
    **Main QA endpoint.** Answers a biomedical question by querying all three literature
    collections (MEDLINE, PMC, Plazi) and returning results ranked by relevance.

    ### How it works

    1. Each collection is searched at the `depth` asked (`fast` by default).
    2. A cross-encoder scores every candidate against the question; the best five are kept.
    3. The five are read by the language model (generative) or BioBERT (extractive).
    4. Results from all three collections are returned in a single response, sorted so
       the most relevant collection appears first (`rank=1`).

    ### Caching

    Identical requests (same question, mode, depth and sizes) within a 120-second window
    are served from cache; an answer given with a fallback note is kept only 10 seconds.

    ### Batch jobs

    A POST that does not come from a web browser (a script) is a batch job, recognised
    automatically: its answers are written at low priority on the shared language-model
    server and may take up to about 100 s. If the model still does not answer, the
    response carries `error` and no answers, never BioBERT's span in place of the
    generated answer. Requests from a browser keep the 4-second fallback.

    ### Direct-document QA

    Set `doc_refs` to a list of PMIDs, PMCIDs, or DOIs to bypass retrieval entirely and
    run QA on specific documents. Unresolved references are reported in `unresolved_refs`.
    """
    batch = _is_batch(http_request)
    try:
        mode, retrieval = _resolve_mode(request.mode, request.retrieval)
        if request.doc_refs:
            result = _run_doc_refs_qa(
                question=request.question,
                doc_refs=request.doc_refs,
                mode=mode,
                debug=request.debug,
                include_documents=request.include_documents,
                batch=batch,
            )
            _log_doc_refs("POST /qa", mode, batch, result)
            return MultiQAResponse(**result)
        p = get_pipeline()
        result = _get_or_run_multi(
            p,
            question=request.question,
            mode=mode,
            retrieval=retrieval,
            retrieval_n=_clamp_retrieval_n(request.retrieval_n),
            final_n=request.final_n,
            debug=request.debug,
            depth=request.depth,
            batch=batch,
        )
        _log_request("POST /qa", mode=mode, batch=batch, mode_used=result.get("mode_used"),
                     depth=request.depth or p.config.default_depth, depth_used=result.get("_depth_used"),
                     notes=result.get("_notes"), n_collections=len(result.get("collection_results", [])),
                     seconds=result.get("pipeline_time"), timings=result.get("_timings"), success=True)
        return MultiQAResponse(**_public(result))
    except Exception as e:
        logger.exception("Pipeline error in /qa")
        _log_request("POST /qa", mode=request.mode, batch=batch, depth=request.depth, success=False,
                     error=type(e).__name__)
        # Report the failure. Returning an empty result with no error made a broken
        # request indistinguishable from a question with no matching literature, which
        # is how 102 deterministic failures went unexplained in the D10.3 evaluation.
        return MultiQAResponse(
            question=request.question,
            collection_results=[],
            mode_used=request.mode,
            ndocs_retrieved=0,
            model="",
            pipeline_time=None,
            error=f"{type(e).__name__}: {e}",
        )


@app.post(
    "/qa/multi",
    tags=["Question Answering"],
    summary="Answer a question across all collections (explicit multi endpoint)",
    response_model=MultiQAResponse,
    response_description="Answers from all three collections, ranked by relevance",
)
def answer_question_multi(request: QuestionRequest, http_request: Request):
    """
    Identical to `POST /qa` but always runs the full multi-collection pipeline without
    the shared cache. Use this endpoint when you need a fresh result, or for direct
    integration that does not rely on cache deduplication.

    Unlike `POST /qa`, concurrent calls to this endpoint will each trigger an independent
    pipeline run (including a separate vLLM generation). For most use cases, `POST /qa`
    is preferred.

    Supports `doc_refs` for direct-document QA in the same way as `POST /qa`, and treats
    a script's request as a batch job in the same way.
    """
    batch = _is_batch(http_request)
    try:
        mode, retrieval = _resolve_mode(request.mode, request.retrieval)
        if request.doc_refs:
            result = _run_doc_refs_qa(
                question=request.question,
                doc_refs=request.doc_refs,
                mode=mode,
                debug=request.debug,
                include_documents=request.include_documents,
                batch=batch,
            )
            _log_doc_refs("POST /qa/multi", mode, batch, result)
            return MultiQAResponse(**result)
        p = get_pipeline()
        result = p.run_multi_collection(
            question=request.question,
            retrieval_n=_clamp_retrieval_n(request.retrieval_n),
            final_n=request.final_n,
            mode=mode,
            retrieval=retrieval,
            debug=request.debug,
            depth=request.depth,
            batch=batch,
        )
        _log_request("POST /qa/multi", mode=mode, batch=batch, mode_used=result.get("mode_used"),
                     depth=request.depth or p.config.default_depth, depth_used=result.get("_depth_used"),
                     notes=result.get("_notes"), n_collections=len(result.get("collection_results", [])),
                     seconds=result.get("pipeline_time"), timings=result.get("_timings"), success=True)
        return MultiQAResponse(**_public(result))
    except Exception as e:
        logger.exception("Pipeline error in /qa/multi")
        _log_request("POST /qa/multi", mode=request.mode, batch=batch, depth=request.depth, success=False,
                     error=type(e).__name__)
        return MultiQAResponse(
            question=request.question,
            collection_results=[],
            mode_used=request.mode,
            ndocs_retrieved=0,
            model="",
            pipeline_time=None,
            error=f"{type(e).__name__}: {e}",
        )


@app.get(
    "/QA",
    tags=["Question Answering"],
    summary="Answer a question — legacy GET endpoint",
    response_model=QAResponse,
    response_description="Single-collection answer, backwards-compatible with the old SIBILS QA API",
)
def answer_question_get(
    q: str = Query(..., description="Natural-language biomedical question."),
    col: Optional[str] = Query(
        default=None,
        description=(
            'Collection to search. One of `"medline"`, `"pmc"`, or `"plazi"`. '
            "Omit to search all collections (results merged)."
        ),
    ),
    n: Optional[int] = Query(
        default=None,
        description=(
            "Number of documents the reader gets and the response cites, at most 5 (default: 5)."
        ),
    ),
    mode: str = Query(
        default="extractive",
        description=(
            'Answer mode: `"extractive"` (BioBERT span, default) or `"generative"` (the language '
            'model). When the model is slow or down, a generative request gets the extractive '
            'answer, with `mode_used: "extractive"` and a note. `"hybrid"` = `generative`.'
        ),
    ),
    depth: Optional[Literal["fast", "deep"]] = Query(
        default=None,
        description=(
            '`"fast"` (default, about 3 s): two keyword searches, merged and reranked. '
            '`"deep"` (about 5 s): also searches written by the language model. '
            '`depth_used` says what ran.'
        ),
    ),
    retrieval: Optional[str] = Query(
        default=None,
        description="Ignored. Kept so that older clients sending `sparse` or `dense` still work; use `depth`.",
    ),
):
    """
    Legacy GET endpoint, backwards-compatible with `biodiversitypmc.sibils.org/api/QA`.

    Returns a single `QAResponse` for one collection rather than the ranked multi-collection
    format of `POST /qa`. Prefer `POST /qa` for new integrations.

    **Example:**
    ```
    GET /api/QA?col=medline&q=What+causes+malaria%3F&n=5&mode=generative&depth=fast
    ```
    """
    return _run_qa(question=q, col=col, n=n, mode=mode, depth=depth)


# ---------------------------------------------------------------------------
# Document fetch (proxy to SIBILS)
# ---------------------------------------------------------------------------

@app.get(
    "/fetch",
    tags=["Documents"],
    summary="Fetch a document — proxy to the SIBILS fetch API",
    response_description="Raw SIBILS fetch response (JSON), passed through unchanged",
)
def fetch_document(request: Request):
    """
    Same-origin proxy for the SIBILS document fetch API.

    The document viewer (PMCA) loads full annotated documents from the same origin as
    the app to avoid CORS. The app is served under `/api`, so the viewer calls
    `GET /api/fetch?ids=...&col=...&format=...`. This endpoint forwards those query
    parameters unchanged to the SIBILS fetch API and returns its response verbatim.

    **Example:** `GET /api/fetch?col=pmc&format=PAM&ids=PMC4248671`
    """
    params = dict(request.query_params)
    if not params.get("ids"):
        raise HTTPException(status_code=400, detail="Missing required query parameter 'ids'")

    # Derive the fetch URL from the configured SIBILS search URL (…/api/search → …/api/fetch)
    base = get_config().sibils.search_api_url.rsplit("/", 1)[0]
    fetch_url = f"{base}/fetch"
    try:
        upstream = requests.get(fetch_url, params=params, timeout=60)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"SIBILS fetch failed: {exc}")

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


# ---------------------------------------------------------------------------
# Batch endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/batch",
    tags=["Batch"],
    summary="Answer multiple questions in parallel",
    response_description="List of QA results in the same order as the input questions",
)
def answer_batch(request: BatchRequest):
    """
    Answers multiple questions in a single request.

    Searches (fast depth) run concurrently across all questions using a thread pool,
    which significantly reduces wall-clock time compared to sequential calls.
    Generation is sequential (one vLLM call per question).

    Results are returned in the same order as the input `questions` list.

    **Tip:** This endpoint always uses extractive mode and the fast depth. For generative
    answers over a list of questions, call `POST /qa` once per question.
    """
    p = get_pipeline()
    t0 = _time.time()
    results = p.run_batch(
        questions=request.questions,
        retrieval_n=request.retrieval_n,
        final_n=request.final_n,
        collection=request.col,
    )
    _log_request("POST /batch", collection=request.col, mode="extractive", depth="fast",
                 n_questions=len(results), seconds=round(_time.time() - t0, 3), success=True)
    return {"results": results, "count": len(results)}


def main():
    """Entry point for the BioMoQA API server."""
    import uvicorn

    host = config.server.host
    port = config.server.port
    workers = config.server.workers
    log_level = config.server.log_level.lower()

    logger.info(f"Starting server on {host}:{port} with {workers} worker(s)")

    uvicorn.run(
        "src.biomoqa_rag.api_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
