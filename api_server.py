#!/usr/bin/env python3
"""
FastAPI Server for BioMoQA RAG
Serves RAG endpoint at http://egaillac.lan.text-analytics.ch:9000

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 9000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import time

from src.pipeline_vllm import FastRAGPipeline, RAGConfig

# Initialize FastAPI
app = FastAPI(
    title="BioMoQA RAG API",
    description="Biomedical Question Answering with RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
)

# Global pipeline instance (loaded on startup)
pipeline = None


class QuestionRequest(BaseModel):
    """Request model for QA endpoint."""
    question: str
    retrieval_n: Optional[int] = 50
    rerank_n: Optional[int] = 20
    include_documents: Optional[bool] = False


class CitationDetail(BaseModel):
    """Detailed citation information."""
    document_id: int
    document_title: str
    pmcid: str


class Answer(BaseModel):
    """Answer with citations."""
    text: str
    citation_ids: List[int]  # Document IDs that support this sentence
    citations: List[CitationDetail]  # Full citation details


class QuestionResponse(BaseModel):
    """Response model for QA endpoint."""
    question: str
    answer: List[Answer]
    references: List[str]
    response_length: int
    pipeline_time: float
    num_retrieved: int
    documents: Optional[List[Dict]] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup (takes ~30s)."""
    global pipeline

    print("="*80)
    print("Loading BioMoQA RAG Pipeline...")
    print("="*80)

    config = RAGConfig(
        retrieval_n=50,
        rerank_n=20,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_vllm=True,  # Fast inference
        gpu_memory_utilization=0.8,
    )

    pipeline = FastRAGPipeline(config)

    print("\n✓ Pipeline loaded and ready!")
    print("=" * 80)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "BioMoQA RAG API",
        "version": "1.0.0",
        "description": "Biomedical Question Answering with Retrieval-Augmented Generation",
        "endpoints": {
            "/qa": "POST - Ask a biomedical question",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation",
        },
        "example": {
            "endpoint": "/qa",
            "method": "POST",
            "body": {
                "question": "What is the host of Plasmodium falciparum?",
                "retrieval_n": 50,
                "rerank_n": 20,
            }
        },
        "citation_explanation": {
            "citation_ids": "List of document IDs (0-based index) that support each sentence",
            "citations": "Full citation details with document title and PMCID",
            "references": "All documents retrieved, indexed by position"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "message": "Pipeline is still loading"}
        )

    return {
        "status": "healthy",
        "model": pipeline.config.model_name,
        "ready": True,
    }


@app.post("/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a biomedical question using RAG.

    Args:
        question: The biomedical question to answer
        retrieval_n: Number of documents to retrieve (default: 50)
        rerank_n: Number of documents to use for generation (default: 20)
        include_documents: Include retrieved documents in response (default: False)

    Returns:
        Answer with sentence-level citations and metadata
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is still loading. Please try again in a moment."
        )

    try:
        # Run RAG pipeline
        result = pipeline.run(
            question=request.question,
            retrieval_n=request.retrieval_n or 50,
            rerank_n=request.rerank_n or 20,
            return_documents=request.include_documents,
        )

        return QuestionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    return {
        "model": pipeline.config.model_name,
        "retrieval_collection": pipeline.config.retrieval_collection,
        "retrieval_n": pipeline.config.retrieval_n,
        "rerank_n": pipeline.config.rerank_n,
        "using_vllm": pipeline.config.use_vllm,
    }


# For development/testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
