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

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# MIG / vGPU workaround for sibils-prod-ai (GRID A100D-1-20C in MIG mode):
#
# 1. CUDA_VISIBLE_DEVICES must be the MIG UUID so cuda:0 maps to the MIG partition.
#    But vLLM 0.17 tries to int() that string during config creation → pydantic error.
#    Fix: warm-up CUDA with the UUID first, then reset to "0" (vLLM-parseable).
#    CUDA keeps the context bound to the MIG device for the process lifetime.
#
# 2. PyTorch 2.4+ enables expandable_segments (VMM via cuMemCreate) by default.
#    VMM is NOT supported on NVIDIA vGPU/GRID → "CUDA driver error: operation not supported".
#    Fix: disable expandable_segments before the first CUDA call.
_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cuda_vis.startswith("MIG-"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    import torch
    torch.zeros(1, device="cuda:0")  # init CUDA context with MIG partition
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # reset to numeric for vLLM config

import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict

from .config import get_config
from .pipeline import RAGPipeline, RAGConfig

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
            logger.info("Mode: GPU (default)")
            rag_config = RAGConfig(
                retrieval_n=config.retrieval.retrieval_n,
                use_smart_retrieval=config.retrieval.use_smart_retrieval,
                hybrid_alpha=config.retrieval.hybrid_alpha,
                use_reranking=config.reranking.enabled,
                reranker_model=config.reranking.model,
                rerank_n=config.reranking.top_k,
                use_relevance_filter=config.relevance_filter.enabled,
                final_n=config.relevance_filter.final_n,
                max_tokens=config.generation.max_tokens,
                temperature=config.generation.temperature,
                max_abstract_length=config.context.max_abstract_length,
                truncate_abstracts=config.context.truncate_abstracts,
                quantization=config.model.quantization,
                gpu_memory_utilization=config.model.gpu_memory_utilization,
            )

        pipeline = RAGPipeline(rag_config)
        logger.info("Pipeline ready")
    return pipeline


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    retrieval_n: Optional[int] = None  # None = use config default (15)
    final_n: Optional[int] = None      # None = use config default (5)
    include_documents: bool = True
    debug: bool = False

    model_config = {"json_schema_extra": {"example": {
        "question": "What causes malaria?",
        "retrieval_n": 30,
        "final_n": 5,
        "include_documents": True,
        "debug": False,
    }}}


class CitationDetail(BaseModel):
    """Citation information for a single document"""
    document_id: int
    document_title: str
    pmcid: str


class AnswerSentence(BaseModel):
    """Single sentence with citations"""
    text: str
    citation_ids: List[int]
    citations: List[CitationDetail]


class QuestionResponse(BaseModel):
    """Ragnarok-style response with sentence-level citations"""
    question: str
    answer: List[AnswerSentence]  # List of sentences with citations
    references: List[str]  # List of all cited documents
    response_length: int
    pipeline_time: float
    num_retrieved: int
    pipeline_version: str
    debug_info: Optional[Dict] = None
    documents: Optional[List[Dict]] = None


@app.on_event("startup")
def startup_event():
    """Initialize pipeline on startup"""
    logger.info("=" * 60)
    logger.info("Starting BioMoQA RAG API")
    logger.info("=" * 60)
    get_pipeline()


@app.get("/health")
def health_check():
    """Health check endpoint"""
    config_info = {}
    if pipeline is not None:
        config_info = {
            "model": pipeline.config.model_name,
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


@app.post("/qa", response_model=QuestionResponse)
def answer_question(
    request: QuestionRequest,
    col: Optional[str] = Query(
        default=None,
        description='SIBILS collection to search: "medline", "plazi", "pmc", or "suppdata". '
                    'If omitted, searches both medline and plazi.',
    ),
):
    """
    Answer a biomedical question using the RAG pipeline.

    Features:
    - Hybrid retrieval (SIBILS + Dense FAISS)
    - Cross-encoder reranking
    - Sentence-level citations

    Use the `col` query parameter to restrict retrieval to a single
    SIBILS collection (e.g. `?col=plazi`).  When omitted the pipeline
    searches **medline + plazi** by default.

    Returns:
        Answer with citations and metadata
    """
    # Validate collection if provided
    from .retrieval.sibils_retriever import SIBILSRetriever
    if col is not None and col not in SIBILSRetriever.VALID_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection '{col}'. "
                   f"Valid values: {', '.join(sorted(SIBILSRetriever.VALID_COLLECTIONS))}",
        )

    try:
        p = get_pipeline()

        result = p.run(
            question=request.question,
            retrieval_n=request.retrieval_n,
            final_n=request.final_n,
            collection=col,
            return_documents=request.include_documents,
            debug=request.debug,
        )

        return QuestionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
