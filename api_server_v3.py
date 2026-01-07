"""
FastAPI server for V3 RAG pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import time

from src.pipeline_vllm_v3 import FastRAGPipelineV3, RAGConfigV3

# Initialize app
app = FastAPI(
    title="BioMoQA RAG API V3",
    description="Ultra-fast biomedical QA with hybrid retrieval",
    version="3.0.0"
)

# Initialize pipeline (lazy loading)
pipeline = None


def get_pipeline():
    """Lazy load pipeline"""
    global pipeline
    if pipeline is None:
        print("Initializing V3 pipeline...")
        config = RAGConfigV3(
            retrieval_n=20,
            use_smart_retrieval=True,
            use_reranking=True,
            final_n=15,
            max_tokens=512
        )
        pipeline = FastRAGPipelineV3(config)
        print("✓ V3 pipeline ready")
    return pipeline


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    retrieval_n: Optional[int] = None
    final_n: Optional[int] = None
    include_documents: bool = False
    debug: bool = False


class QuestionResponse(BaseModel):
    question: str
    answer: str
    num_retrieved: int
    pipeline_time: float
    pipeline_version: str
    debug_info: Optional[Dict] = None
    documents: Optional[List[Dict]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    print("="*80)
    print("Starting BioMoQA RAG API V3")
    print("="*80)
    get_pipeline()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_version": "v3",
        "improvements": {
            "smart_hybrid_retrieval": True,
            "parallel_execution": True,
            "fast_reranking": True,
            "relevance_filtering": True
        },
        "ready": pipeline is not None
    }


@app.post("/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a biomedical question using V3 RAG pipeline.

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


@app.get("/compare")
async def compare_versions():
    """Compare V1, V2, and V3"""
    return {
        "v1": {
            "retrieval": "SIBILS BM25 only",
            "reranking": "None",
            "filtering": "None",
            "query_expansion": "None",
            "avg_time": "7.27s",
            "quality": "Good"
        },
        "v2": {
            "retrieval": "SIBILS BM25 + multi-query",
            "reranking": "Cross-encoder semantic",
            "filtering": "Keyword-based",
            "query_expansion": "LLM + rules",
            "avg_time": "11.19s",
            "quality": "Very Good"
        },
        "v3": {
            "retrieval": "Parallel hybrid (BM25 + Dense with RRF)",
            "reranking": "Fast cross-encoder",
            "filtering": "Optimized keyword",
            "query_expansion": "Disabled (speed)",
            "avg_time": "6.81s",
            "quality": "Very Good",
            "speedup_vs_v2": "+39.1%",
            "speedup_vs_v1": "+6.3%"
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "BioMoQA RAG API",
        "version": "3.0.0",
        "pipeline": "V3 Speed-Optimized Hybrid Retrieval",
        "endpoints": {
            "/health": "Health check",
            "/qa": "Answer questions (POST)",
            "/compare": "Compare V1 vs V2 vs V3",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
