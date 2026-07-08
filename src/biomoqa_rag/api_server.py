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

import logging
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

import requests
import threading
import time as _time
from .config import get_config
from .pipeline import RAGPipeline, RAGConfig


# Per-question cache for multi-collection results.
# The frontend calls /qa 4× in parallel (one per collection tab).
# The first call runs the full pipeline and caches; the other 3 wait and
# return from cache immediately, avoiding 4× sequential vLLM generations.
_multi_cache: dict = {}           # (question, mode) -> (result, timestamp)
_multi_in_flight: dict = {}       # (question, mode) -> threading.Event
_multi_cache_mutex = threading.Lock()
_MULTI_CACHE_TTL = 120            # seconds


def _get_or_run_multi(pipeline, question, mode, retrieval, retrieval_n, final_n, debug):
    key = (question, mode, retrieval, retrieval_n, final_n)
    with _multi_cache_mutex:
        if key in _multi_cache:
            result, ts = _multi_cache[key]
            if _time.time() - ts < _MULTI_CACHE_TTL:
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
                final_n=final_n, mode=mode, retrieval=retrieval, debug=debug,
            )
            with _multi_cache_mutex:
                _multi_cache[key] = (result, _time.time())
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

# ---------------------------------------------------------------------------
# OpenAPI metadata
# ---------------------------------------------------------------------------

_APP_DESCRIPTION = """
## BioMoQA RAG — Biomedical Question Answering

Answers natural-language biomedical questions by combining **retrieval** from three literature
collections with **LLM-based generation** (Qwen3-8B via vLLM) or **extractive span detection**
(BioBERT).

---

### Pipeline

```
Question
  └─▶ Retrieval  ── SIBILS BM25 (sparse)  ─┐
                 └─ FAISS dense + reranker ─┴─▶ top-k docs
       └─▶ Relevance filter
             └─▶ Generation (Qwen3-8B) or Extraction (BioBERT)
                   └─▶ Answer + sentence-level citations
```

---

### Collections

| ID | Source | Content |
|----|--------|---------|
| `medline` | PubMed / MEDLINE | Biomedical abstracts |
| `pmc` | PubMed Central | Full-text open-access articles |
| `plazi` | Plazi | Biodiversity & taxonomy literature |

All three collections are queried by default. Restrict with `col` (GET endpoint) or use
`doc_refs` to bypass retrieval entirely and target specific documents.

---

### Answer modes

| `mode` | Model | Output |
|--------|-------|--------|
| `generative` | Qwen3-8B (GPU, vLLM) | Fluent answer with inline `[N]` sentence-level citations |
| `extractive` | BioBERT (CPU/GPU) | Verbatim span lifted from the best matching document + confidence score |

---

### Retrieval strategies

| `retrieval` | Method | Best for |
|-------------|--------|----------|
| `sparse` | SIBILS BM25 keyword search | Exact terms, gene/drug names, acronyms |
| `dense` | BM25 + FAISS semantic search + cross-encoder reranker | Conceptual questions, synonyms, paraphrases |

Legacy aliases: `mode=hybrid` → `generative + dense`; `retrieval=rag` → `dense`.

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

        mode = config.model.mode
        model_size = config.model.size

        if mode == "cpu":
            logger.info(f"Mode: CPU inference, model size: {model_size}")
            rag_config = RAGConfig.cpu_config(model_size=model_size)
        elif mode == "gpu_small":
            logger.info("Mode: GPU (small model, ~8GB VRAM)")
            rag_config = RAGConfig.gpu_small_config()
        else:
            logger.info(f"Mode: GPU (default), model: {config.model.model_name}")
            rag_config = RAGConfig(
                model_name=config.model.model_name,
                retrieval_n=config.retrieval.retrieval_n,
                use_smart_retrieval=config.retrieval.use_smart_retrieval,
                hybrid_alpha=config.retrieval.hybrid_alpha,
                use_reranking=config.reranking.enabled,
                reranker_model=config.reranking.model,
                rerank_n=config.reranking.top_k,
                use_relevance_filter=config.relevance_filter.enabled,
                final_n=config.relevance_filter.final_n,
                min_extractive_score=config.relevance_filter.min_extractive_score,
                max_tokens=config.generation.max_tokens,
                temperature=config.generation.temperature,
                max_abstract_length=config.context.max_abstract_length,
                llm_abstract_length=config.context.llm_abstract_length,
                truncate_abstracts=config.context.truncate_abstracts,
                quantization=config.model.quantization,
                gpu_memory_utilization=config.model.gpu_memory_utilization,
                qa_device=config.extraction.device,
                sibils_cache_dir=config.sibils.cache_dir,
                sibils_cache_ttl=config.sibils.cache_ttl,
                sibils_empty_cache_ttl=config.sibils.empty_cache_ttl,
                sibils_collections=config.sibils.collections,
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
            "Number of documents fetched from SIBILS before reranking (default: 10, max: 100). "
            "Higher values improve recall at the cost of latency (~0.2 s per 10 extra docs)."
        ),
    )
    final_n: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of documents kept after relevance filtering and passed to the LLM (default: 5). "
            "Smaller values produce more focused answers; larger values improve coverage."
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
            "- `generative` *(default)* — Qwen3-8B generates a fluent answer with inline `[N]` citations\n"
            "- `extractive` — BioBERT extracts a verbatim span from the best matching document "
            "and returns a confidence score\n"
            "- `hybrid` *(legacy alias)* — mapped to `generative` + `dense` retrieval"
        ),
    )
    retrieval: str = Field(
        default="sparse",
        description=(
            "Retrieval strategy:\n"
            "- `sparse` *(default)* — SIBILS BM25 keyword search; fast, best for exact terms and acronyms\n"
            "- `dense` — BM25 + FAISS semantic search fused with RRF, then re-ranked by a cross-encoder; "
            "better recall for conceptual or paraphrased questions\n"
            "- `rag` *(legacy alias)* — mapped to `dense`"
        ),
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
        "retrieval": "dense",
        "retrieval_n": 20,
        "final_n": 5,
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
        description="Retrieval or reranking relevance score (higher = more relevant to the question).",
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


class AnswerItem(BaseModel):
    """A single answer candidate with its supporting documents."""

    answer: str = Field(
        ...,
        description=(
            "The answer text. "
            "In **generative** mode: a fluent sentence synthesised by the LLM, with inline `[N]` "
            "markers that correspond to entries in `docs`. "
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
    docs: List[DocItem] = Field(
        default_factory=list,
        description=(
            "Supporting documents for this answer. "
            "Extractive: exactly one item with `snippet_start`/`snippet_end` offsets. "
            "Generative: one entry per sentence cited (matching the `[N]` markers in `answer`)."
        ),
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
            "Relevance rank of this collection for the question (1 = best match). "
            "Rank is determined by the score of the top retrieved document."
        ),
    )
    answers: List[AnswerItem] = Field(
        ...,
        description="Answer candidates for this collection, ordered best-first.",
    )
    mode_used: Optional[str] = Field(
        None,
        description="Effective mode after legacy-alias resolution, e.g. `generative` or `extractive`.",
    )
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
            "`biobert` for extractive mode, or the generative model name (e.g. `Qwen3-8B-FP8`)."
        ),
    )
    pipeline_time: Optional[float] = Field(
        None,
        description="Total wall-clock time for the full pipeline in seconds.",
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
    ndocs_requested: int = Field(..., description="Number of documents requested from SIBILS.")
    ndocs_returned_by_SIBiLS: int = Field(..., description="Number of documents actually returned by SIBILS.")
    answers: List[AnswerItem] = Field(..., description="Ranked answer candidates, best first.")
    mode_used: Optional[str] = Field(None, description="Effective mode after alias resolution.")
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
    Returns the service status and, once the pipeline is initialised, the names of the
    loaded models and the inference mode (CPU / GPU).

    The `ready` field is `false` during startup (pipeline not yet loaded) and `true`
    once the first request has triggered pipeline initialisation.
    """
    config_info = {}
    if pipeline is not None:
        config_info = {
            "generative_model": pipeline.config.model_name,
            "extractive_model": pipeline.config.qa_model,
            "inference_mode": "cpu" if pipeline.config.use_cpu else "gpu",
            "use_vllm": pipeline.config.use_vllm,
        }

    return {
        "status": "healthy",
        "features": {
            "hybrid_retrieval": True,
            "cross_encoder_reranking": True,
            "relevance_filtering": True,
            "sentence_citations": True,
            "cpu_inference": True,
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
    Returns a human-readable description of the hybrid retrieval system: BM25 via SIBILS,
    dense FAISS search, Reciprocal Rank Fusion, and the cross-encoder reranker.

    Useful for understanding the tradeoffs between `sparse` and `dense` retrieval modes.
    """
    return {
        "hybrid_retrieval": {
            "description": "Combines SIBILS (BM25) and Dense (FAISS) retrieval",
            "sibils_bm25": {
                "description": "Keyword-based search via SIBILS API",
                "corpus": "10,000+ PMC biomedical papers",
                "speed": "~1.9s per query",
                "best_for": "Exact terms, acronyms, technical queries"
            },
            "dense_faiss": {
                "description": "Semantic vector search with local FAISS index",
                "corpus": "2,398 biomedical documents",
                "speed": "~0.07s per query (96% faster than SIBILS!)",
                "best_for": "Semantic meaning, paraphrases, conceptual queries",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "fusion": {
                "method": "Reciprocal Rank Fusion (RRF)",
                "execution": "Parallel (both run simultaneously)",
                "result": "Best documents from both sources combined"
            },
            "smart_strategy": {
                "technical_query": "Uses SIBILS only (e.g., 'What is AG1-IA?')",
                "semantic_query": "Uses Dense only - 96% faster! (e.g., 'How does immune system work?')",
                "general_query": "Uses both in parallel (e.g., 'What causes malaria?')"
            }
        },
        "why_still_use_sibils": [
            "SIBILS has 10,000+ papers (vs 2,398 in local index)",
            "Best for exact medical terms and acronyms",
            "Complements semantic search for comprehensive coverage",
            "Running in parallel means no added time cost"
        ],
        "example": {
            "query": "How does the immune system fight viral infections?",
            "sibils_finds": "Papers with 'immune system', 'viral infections'",
            "dense_finds": "Papers about 'host defense', 'antiviral response' (semantic)",
            "combined": "Best coverage from both approaches"
        }
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

    collection_results = []
    for rank, (col, col_docs) in enumerate(sorted(by_col.items()), start=1):
        result = p.run_with_documents(
            question=question,
            documents=col_docs,
            mode=mode,
            return_documents=include_documents,
            debug=debug,
        )
        collection_results.append({
            "collection": col,
            "rank": rank,
            "answers": result.get("answers", []),
            "mode_used": result.get("mode_used"),
            "debug_info": result.get("debug_info") if debug else None,
        })

    model_name = (
        "biobert" if mode == "extractive"
        else p.config.model_name.split("/")[-1]
    )
    return {
        "question": question,
        "collection_results": collection_results,
        "mode_used": mode,
        "ndocs_retrieved": len(docs),
        "model": model_name,
        "pipeline_time": round(_t.time() - t0, 3),
        "unresolved_refs": unresolved if unresolved else None,
    }


def _clamp_retrieval_n(n: Optional[int]) -> Optional[int]:
    if n is None:
        return None
    return max(1, min(n, _MAX_RETRIEVAL_N))


def _resolve_mode(mode: str, retrieval: str) -> tuple[str, str]:
    """Normalise mode/retrieval values; map legacy names to current ones."""
    # Legacy mode aliases
    if mode == "hybrid":
        return "generative", "dense"
    # Legacy retrieval aliases
    if retrieval == "rag":
        retrieval = "dense"
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
) -> QAResponse:
    """Shared logic for both POST /qa and GET /QA."""
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
        result = p.run(
            question=question,
            retrieval_n=retrieval_n or n,
            final_n=final_n,
            collection=col,
            return_documents=include_documents,
            debug=debug,
            mode=mode,
        )
        return QAResponse(**result)
    except Exception as e:
        logger.exception("Pipeline error")
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
def answer_question_post(request: QuestionRequest):
    """
    **Main QA endpoint.** Answers a biomedical question by querying all three literature
    collections (MEDLINE, PMC, Plazi) and returning results ranked by relevance.

    ### How it works

    1. Documents are retrieved from SIBILS (BM25) and optionally from the local FAISS
       index, depending on the `retrieval` strategy.
    2. Retrieved documents are reranked by a cross-encoder and filtered by relevance.
    3. The top documents are passed to either Qwen3-8B (generative) or BioBERT (extractive).
    4. Results from all three collections are returned in a single response, sorted so
       the most relevant collection appears first (`rank=1`).

    ### Caching

    Identical requests (same question, mode, and retrieval parameters) within a 120-second
    window are served from cache. This avoids redundant LLM calls when the frontend polls
    multiple collection tabs simultaneously.

    ### Direct-document QA

    Set `doc_refs` to a list of PMIDs, PMCIDs, or DOIs to bypass retrieval entirely and
    run QA on specific documents. Unresolved references are reported in `unresolved_refs`.
    """
    try:
        mode, retrieval = _resolve_mode(request.mode, request.retrieval)
        if request.doc_refs:
            result = _run_doc_refs_qa(
                question=request.question,
                doc_refs=request.doc_refs,
                mode=mode,
                debug=request.debug,
                include_documents=request.include_documents,
            )
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
        )
        return MultiQAResponse(**result)
    except Exception as e:
        logger.exception("Pipeline error in /qa")
        return MultiQAResponse(
            question=request.question,
            collection_results=[],
            mode_used=request.mode,
            ndocs_retrieved=0,
            model="",
            pipeline_time=None,
        )


@app.post(
    "/qa/multi",
    tags=["Question Answering"],
    summary="Answer a question across all collections (explicit multi endpoint)",
    response_model=MultiQAResponse,
    response_description="Answers from all three collections, ranked by relevance",
)
def answer_question_multi(request: QuestionRequest):
    """
    Identical to `POST /qa` but always runs the full multi-collection pipeline without
    the shared cache. Use this endpoint when you need a fresh result, or for direct
    integration that does not rely on cache deduplication.

    Unlike `POST /qa`, concurrent calls to this endpoint will each trigger an independent
    pipeline run (including a separate vLLM generation). For most use cases, `POST /qa`
    is preferred.

    Supports `doc_refs` for direct-document QA in the same way as `POST /qa`.
    """
    try:
        mode, retrieval = _resolve_mode(request.mode, request.retrieval)
        if request.doc_refs:
            result = _run_doc_refs_qa(
                question=request.question,
                doc_refs=request.doc_refs,
                mode=mode,
                debug=request.debug,
                include_documents=request.include_documents,
            )
            return MultiQAResponse(**result)
        p = get_pipeline()
        result = p.run_multi_collection(
            question=request.question,
            retrieval_n=_clamp_retrieval_n(request.retrieval_n),
            final_n=request.final_n,
            mode=mode,
            retrieval=retrieval,
            debug=request.debug,
        )
        return MultiQAResponse(**result)
    except Exception as e:
        logger.exception("Pipeline error in /qa/multi")
        return MultiQAResponse(
            question=request.question,
            collection_results=[],
            mode_used=request.mode,
            ndocs_retrieved=0,
            model="",
            pipeline_time=None,
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
            "Number of documents to retrieve from SIBILS (default: 10). "
            "Higher values improve recall but increase latency."
        ),
    ),
    mode: str = Query(
        default="extractive",
        description=(
            'Answer mode: `"extractive"` (BioBERT span, default) or `"generative"` (Qwen3-8B). '
            '`"hybrid"` is a legacy alias for `generative` + dense retrieval.'
        ),
    ),
):
    """
    Legacy GET endpoint, backwards-compatible with `biodiversitypmc.sibils.org/api/QA`.

    Returns a single `QAResponse` for one collection rather than the ranked multi-collection
    format of `POST /qa`. Prefer `POST /qa` for new integrations.

    **Example:**
    ```
    GET /api/QA?col=medline&q=What+causes+malaria%3F&n=5&mode=generative
    ```
    """
    return _run_qa(question=q, col=col, n=n, mode=mode)


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

    Retrieval (SIBILS + FAISS) runs concurrently across all questions using a thread pool,
    which significantly reduces wall-clock time compared to sequential calls.
    Generation is sequential (one vLLM call per question).

    Results are returned in the same order as the input `questions` list.

    **Tip:** This endpoint always uses extractive mode for speed. For generative answers
    over a list of questions, call `POST /qa` once per question.
    """
    p = get_pipeline()
    results = p.run_batch(
        questions=request.questions,
        retrieval_n=request.retrieval_n,
        final_n=request.final_n,
        collection=request.col,
    )
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
