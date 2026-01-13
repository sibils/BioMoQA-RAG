"""
FastAPI server for V3.2 RAG pipeline with SIBILS query parser integration.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from pipeline_vllm_v3_fast import UltraFastRAGPipeline, RAGConfigV3Fast

# Initialize app
app = FastAPI(
    title="BioMoQA RAG API V3.2",
    description="Biomedical QA with SIBILS query parser, hybrid retrieval, and sentence-level citations",
    version="3.2.0"
)

# Initialize pipeline (lazy loading)
pipeline = None


def get_pipeline():
    """Lazy load pipeline"""
    global pipeline
    if pipeline is None:
        print("Initializing V3.2 pipeline with query parser...")
        config = RAGConfigV3Fast(
            retrieval_n=20,
            use_smart_retrieval=True,
            use_reranking=True,
            final_n=10,
            max_tokens=384,
            truncate_abstracts=True,
            quantization=None,  # Disabled to avoid GPU memory issues
            gpu_memory_utilization=0.4  # Reduced for MIG GPU (76GB available, need <40GB)
        )
        pipeline = UltraFastRAGPipeline(config)
        print("✓ V3.2 pipeline ready")
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
    print("Starting BioMoQA RAG API V3.2")
    print("="*80)
    get_pipeline()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_version": "v3.2",
        "optimizations": {
            "query_parser_es_mode": True,
            "concept_expansion": True,
            "smart_hybrid_retrieval": True,
            "parallel_execution": True,
            "cross_encoder_reranking": True,
            "relevance_filtering": True,
            "sentence_citations": True
        },
        "expected_speed": "~7s per question",
        "ready": pipeline is not None
    }


@app.post("/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a biomedical question using V3.2 ultra-fast RAG pipeline.

    Features:
    - SIBILS query parser generates Elasticsearch queries with concept expansion
    - Hybrid retrieval (Elasticsearch + Dense FAISS)
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


@app.get("/compare")
async def compare_versions():
    """Compare all versions"""
    return {
        "v1": {
            "retrieval": "SIBILS BM25 only",
            "quantization": "None",
            "avg_time": "7.27s",
            "speedup": "Baseline (24x vs original)",
            "quality": "Good"
        },
        "v2": {
            "retrieval": "SIBILS BM25 + multi-query",
            "reranking": "Cross-encoder",
            "quantization": "None",
            "avg_time": "11.19s",
            "speedup": "+54% slower than V1",
            "quality": "Very Good"
        },
        "v3": {
            "retrieval": "Parallel hybrid (SIBILS + Dense FAISS)",
            "reranking": "Fast cross-encoder",
            "quantization": "None",
            "avg_time": "6.81s",
            "speedup": "+39% faster than V2, +6% faster than V1",
            "quality": "Very Good"
        },
        "v3.1": {
            "retrieval": "Smart hybrid (adaptive SIBILS/Dense/Both)",
            "reranking": "Fast cross-encoder",
            "quantization": "FP8 (33% generation speedup)",
            "context": "10 docs, truncated abstracts",
            "avg_time": "5.20s",
            "speedup": "+24% faster than V3, +54% faster than V2, +29% faster than V1",
            "quality": "Very Good",
            "gpu_memory": "8.1 GB (vs 14.2 GB)"
        }
    }


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
        "version": "3.1.0 (Ultra-Fast)",
        "pipeline": "V3.1 with FP8 Quantization",
        "speed": "~5.2s per question",
        "retrieval": "Smart Hybrid (SIBILS + Dense FAISS)",
        "endpoints": {
            "/health": "Health check",
            "/qa": "Answer questions (POST)",
            "/compare": "Compare all versions",
            "/retrieval-info": "Explain hybrid retrieval",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
