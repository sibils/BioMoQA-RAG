"""
FastAPI server for BioMoQA RAG pipeline.

Configuration is read from config.toml in the working directory.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from .config import get_config
from .pipeline import RAGPipeline, RAGConfig

# Load configuration
config = get_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.get("server", "log_level", default="info").upper()),
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

        mode = config.get("model", "mode", default="gpu")
        model_size = config.get("model", "size", default="3b")

        if mode == "cpu":
            logger.info(f"Mode: CPU inference, model size: {model_size}")
            rag_config = RAGConfig.cpu_config(model_size=model_size)
        elif mode == "gpu_small":
            logger.info("Mode: GPU (small model, ~8GB VRAM)")
            rag_config = RAGConfig.gpu_small_config()
        else:
            logger.info("Mode: GPU (default)")
            rag_config = RAGConfig(
                retrieval_n=config.get("retrieval", "retrieval_n", default=20),
                use_smart_retrieval=config.get("retrieval", "use_smart_retrieval", default=True),
                hybrid_alpha=config.get("retrieval", "hybrid_alpha", default=0.5),
                use_reranking=config.get("reranking", "enabled", default=True),
                reranker_model=config.get("reranking", "model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"),
                rerank_n=config.get("reranking", "top_k", default=15),
                use_relevance_filter=config.get("relevance_filter", "enabled", default=True),
                final_n=config.get("relevance_filter", "final_n", default=10),
                max_tokens=config.get("generation", "max_tokens", default=384),
                temperature=config.get("generation", "temperature", default=0.1),
                max_abstract_length=config.get("context", "max_abstract_length", default=800),
                truncate_abstracts=config.get("context", "truncate_abstracts", default=True),
                quantization=config.get("model", "quantization", default="fp8"),
                gpu_memory_utilization=config.get("model", "gpu_memory_utilization", default=0.8),
            )

        pipeline = RAGPipeline(rag_config)
        logger.info("Pipeline ready")
    return pipeline


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    retrieval_n: Optional[int] = None
    final_n: Optional[int] = None
    include_documents: bool = False
    debug: bool = False


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
async def startup_event():
    """Initialize pipeline on startup"""
    logger.info("=" * 60)
    logger.info("Starting BioMoQA RAG API")
    logger.info("=" * 60)
    get_pipeline()


@app.get("/health")
async def health_check():
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
async def answer_question(request: QuestionRequest):
    """
    Answer a biomedical question using the RAG pipeline.

    Features:
    - Hybrid retrieval (SIBILS + Dense FAISS)
    - Cross-encoder reranking
    - Sentence-level citations

    Returns:
        Answer with citations and metadata
    """
    try:
        p = get_pipeline()

        result = p.run(
            question=request.question,
            retrieval_n=request.retrieval_n,
            final_n=request.final_n,
            return_documents=request.include_documents,
            debug=request.debug
        )

        return QuestionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieval-info")
async def retrieval_info():
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
async def root():
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

    host = config.get("server", "host", default="0.0.0.0")
    port = config.get("server", "port", default=9000)
    workers = config.get("server", "workers", default=1)
    log_level = config.get("server", "log_level", default="info").lower()

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
