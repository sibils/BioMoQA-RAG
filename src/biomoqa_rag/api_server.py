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
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

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
    key = (question, mode, retrieval)
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

# Initialize app
app = FastAPI(
    title="BioMoQA RAG API",
    description="Biomedical question answering with SIBILS retrieval and sentence-level citations",
    version="1.0.0"
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
                sibils_collections=config.sibils.collections,
            )

        pipeline = RAGPipeline(rag_config)
        logger.info("Pipeline ready")
    return pipeline


# ---------------------------------------------------------------------------
# Request / Response models (aligned with biodiversitypmc.sibils.org/api/QA)
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    retrieval_n: Optional[int] = None
    final_n: Optional[int] = None
    include_documents: bool = False
    debug: bool = False
    mode: str = "generative"        # "extractive" | "generative"
    retrieval: str = "elasticsearch"  # "elasticsearch" (BM25 only) | "dense" (FAISS + BM25 + reranker)

    model_config = {"json_schema_extra": {"example": {
        "question": "What causes malaria?",
        "mode": "generative",
        "retrieval": "dense",
    }}}


class AnswerItem(BaseModel):
    """One ranked answer candidate — mirrors the old sibils.org answer object."""
    answer: str
    answer_score: Optional[float] = None    # BioBERT span score; None for generative
    docid: Optional[str] = None             # PMID, Plazi treatment ID, or PMC ID (extractive)
    doc_source: Optional[str] = None        # Collection: "medline", "plazi", "pmc"
    doc_retrieval_score: Optional[float] = None
    doc_text: Optional[str] = None          # Context passage (extractive only)
    snippet_start: Optional[int] = None     # Char offset of answer span in doc_text
    snippet_end: Optional[int] = None       # Char offset of answer span in doc_text
    # Generative-only: list of all cited documents
    docids: Optional[List[str]] = None      # All cited doc IDs
    docs: Optional[List[Dict]] = None       # All cited docs with docid/doc_text/doc_source


class CollectionResult(BaseModel):
    """Per-collection answer block, ordered by relevance rank."""
    collection: str                      # "medline", "plazi", "pmc"
    rank: int                            # 1 = best collection for this question
    answers: List[AnswerItem]
    mode_used: Optional[str] = None
    debug_info: Optional[Dict] = None


class MultiQAResponse(BaseModel):
    """Response for /qa/multi — all collections ranked best-first."""
    question: str
    collection_results: List[CollectionResult]
    mode_used: str
    ndocs_retrieved: int
    model: str
    pipeline_time: Optional[float] = None


class QAResponse(BaseModel):
    """Response format aligned with biodiversitypmc.sibils.org/api/QA."""
    sibils_version: str
    success: bool
    error: str
    question: str
    collection: str
    model: str                          # "biobert" | short generative model name
    ndocs_requested: int
    ndocs_returned_by_SIBiLS: int
    answers: List[AnswerItem]           # Ranked answer candidates (best first)
    # Extra fields not in old API — kept for observability
    mode_used: Optional[str] = None    # e.g. "hybrid:extractive"
    pipeline_time: Optional[float] = None
    transformed_query: None = None     # Not applicable (we use FAISS, not ES)
    debug_info: Optional[Dict] = None
    documents: Optional[List[Dict]] = None


@app.on_event("startup")
def startup_event():
    """Initialize pipeline on startup"""
    logger.info("=" * 60)
    logger.info("Starting BioMoQA RAG API")
    logger.info("=" * 60)
    p = get_pipeline()
    _ = p.extractor  # warm up BioBERT so first request is not slow


@app.get("/health")
def health_check():
    """Health check endpoint"""
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


def _resolve_mode(mode: str, retrieval: str) -> tuple[str, str]:
    """Normalise mode/retrieval values; map legacy names to current ones."""
    # Legacy mode aliases
    if mode == "hybrid":
        return "generative", "dense"
    # Legacy retrieval aliases
    if retrieval == "sibils":
        retrieval = "elasticsearch"
    elif retrieval == "rag":
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
    """Shared logic for both POST /qa and GET /api/QA."""
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


@app.post("/qa", response_model=MultiQAResponse)
def answer_question_post(request: QuestionRequest):
    """
    Answer a biomedical question across all collections (POST).

    Temporarily routes to multi-collection pipeline so the frontend receives
    collection_results in one request instead of 4 separate calls that queue
    behind the vLLM lock. Revert once frontend switches to /qa/multi.
    """
    try:
        p = get_pipeline()
        mode, retrieval = _resolve_mode(request.mode, request.retrieval)
        result = _get_or_run_multi(
            p,
            question=request.question,
            mode=mode,
            retrieval=retrieval,
            retrieval_n=request.retrieval_n,
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


@app.get("/api/QA", response_model=QAResponse)
def answer_question_get(
    q: str = Query(..., description="Natural language question"),
    col: Optional[str] = Query(default=None, description='Collection to search: "medline", "plazi", or "pmc". Defaults to all collections.'),
    n: Optional[int] = Query(default=None, description="Number of documents retrieved initially (default: 10). Higher values improve recall but slow down the response."),
    mode: str = Query(default="extractive", description='Answer mode: "extractive" or "generative". Legacy "hybrid" maps to generative+dense.'),
):
    """
    GET endpoint — backwards-compatible with biodiversitypmc.sibils.org/api/QA.

    Example: /api/QA?col=medline&q=What+causes+malaria%3F&n=5
    """
    return _run_qa(question=q, col=col, n=n, mode=mode)


class BatchRequest(BaseModel):
    questions: List[str]
    retrieval_n: Optional[int] = None
    final_n: Optional[int] = None
    col: Optional[str] = None

    model_config = {"json_schema_extra": {"example": {
        "questions": [
            "What causes malaria?",
            "What diseases are associated with ticks?",
        ],
    }}}


@app.post("/batch")
def answer_batch(request: BatchRequest):
    """
    Answer multiple questions using the extractive model with parallel retrieval.

    Retrieval (SIBILS + FAISS) runs concurrently across all questions.
    Results are returned in the same order as the input questions.
    """
    p = get_pipeline()
    results = p.run_batch(
        questions=request.questions,
        retrieval_n=request.retrieval_n,
        final_n=request.final_n,
        collection=request.col,
    )
    return {"results": results, "count": len(results)}


@app.post("/qa/multi", response_model=MultiQAResponse)
def answer_question_multi(request: QuestionRequest):
    """
    Answer a question across all collections (medline, plazi, pmc), ranked best-first.

    Retrieves from 3 collections in a single call, ranks by best document relevance,
    then generates answers sequentially so the frontend can display the most relevant
    collection on the left.

    This avoids concurrent vLLM calls — generation is always sequential regardless
    of how many frontend tabs are open.
    """
    try:
        p = get_pipeline()
        mode, retrieval = _resolve_mode(request.mode, request.retrieval)
        result = p.run_multi_collection(
            question=request.question,
            retrieval_n=request.retrieval_n,
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


@app.get("/retrieval-info")
def retrieval_info():
    """Explain the hybrid retrieval system"""
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


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "BioMoQA RAG API",
        "description": "Biomedical question answering with SIBILS retrieval",
        "endpoints": {
            "/health": "Health check",
            "/qa": "Answer questions (POST)",
            "/retrieval-info": "Explain hybrid retrieval",
            "/docs": "API documentation"
        }
    }


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
