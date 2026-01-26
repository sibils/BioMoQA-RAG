"""
FastAPI server for BioMoQA RAG pipeline.

Environment variables:
    BIOMOQA_USE_CPU: Set to "true" for CPU inference (default: false)
    BIOMOQA_MODEL_SIZE: Model size for CPU mode: "0.5b", "1.5b", "3b", "7b" (default: "3b")
    BIOMOQA_GPU_SMALL: Set to "true" for small GPU model (~8GB VRAM) (default: false)
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from .pipeline import RAGPipeline, RAGConfig

# Initialize app
app = FastAPI(
    title="BioMoQA RAG API",
    description="Biomedical question answering with SIBILS retrieval and sentence-level citations",
    version="1.0.0"
)

# Initialize pipeline (lazy loading)
pipeline = None


def get_pipeline():
    """Lazy load pipeline based on environment configuration"""
    global pipeline
    if pipeline is None:
        print("Initializing BioMoQA RAG pipeline...")

        # Check environment variables for configuration
        use_cpu = os.getenv("BIOMOQA_USE_CPU", "false").lower() == "true"
        use_gpu_small = os.getenv("BIOMOQA_GPU_SMALL", "false").lower() == "true"
        model_size = os.getenv("BIOMOQA_MODEL_SIZE", "3b")

        if use_cpu:
            # CPU inference with transformers
            print(f"  Mode: CPU inference")
            print(f"  Model size: {model_size}")
            config = RAGConfig.cpu_config(model_size=model_size)
        elif use_gpu_small:
            # Small GPU model (~8GB VRAM)
            print("  Mode: GPU (small model, ~8GB VRAM)")
            config = RAGConfig.gpu_small_config()
        else:
            # Default: full GPU mode
            print("  Mode: GPU (full model)")
            config = RAGConfig(
                retrieval_n=20,
                use_smart_retrieval=True,
                use_reranking=True,
                final_n=10,
                max_tokens=384,
                truncate_abstracts=True,
                quantization=None,  # Disabled to avoid GPU memory issues
                gpu_memory_utilization=0.4  # Reduced for MIG GPU
            )

        pipeline = RAGPipeline(config)
        print("✓ Pipeline ready")
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
    print("="*80)
    print("Starting BioMoQA RAG API")
    print("="*80)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
