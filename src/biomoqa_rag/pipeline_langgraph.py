"""
LangGraph sandbox for BioMoQA RAG pipeline.

Drop-in replacement for RAGPipeline.run() using a LangGraph StateGraph.
Inherits all components from RAGPipeline — nothing in pipeline.py is touched.

Graph topology:
    START
      │
      ▼
    [retrieve]
      │
      ├── "extractive" ──► [extractive] ──► END
      ├── "generative" ──► [generative] ──► END
      └── "__end__"    ──────────────────► END  (empty docs)

Usage:
    from biomoqa_rag.pipeline_langgraph import LangGraphRAGPipeline, RAGConfig
    p = LangGraphRAGPipeline(RAGConfig.cpu_config())
    result = p.run("What causes malaria?", collection="medline")
"""

import time
from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # py<3.8 fallback

from .pipeline import RAGConfig, RAGPipeline, PIPELINE_VERSION


class RAGState(TypedDict):
    # ── Inputs (set at graph entry, never mutated by nodes) ──────────────
    question:         str
    retrieval_n:      int
    final_n:          int
    collection:       Optional[str]
    mode:             str    # "extractive" | "generative"
    retrieval:        str    # "sparse" | "dense"
    return_documents: bool
    debug:            bool
    start_time:       float
    # ── Retrieval node output ─────────────────────────────────────────────
    documents:        List
    num_retrieved:    int
    # ── Answer node output ────────────────────────────────────────────────
    answers:          List[Dict]
    # ── Debug info (populated by nodes when debug=True, else None) ────────
    debug_info:       Optional[Dict[str, Any]]


class LangGraphRAGPipeline(RAGPipeline):
    """
    RAGPipeline subclass that orchestrates run() via a LangGraph StateGraph.

    All retrieval, generation, and helper methods are inherited unchanged.
    Only run() is replaced — run_batch() and run_multi_collection() are untouched.
    """

    def __init__(self, config: RAGConfig = None):
        super().__init__(config)
        self._graph = self._build_graph()
        print("✓ LangGraph StateGraph compiled")

    # ── Graph construction ────────────────────────────────────────────────

    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        builder = StateGraph(RAGState)
        builder.add_node("retrieve",   self._node_retrieve)
        builder.add_node("extractive", self._node_extractive)
        builder.add_node("generative", self._node_generative)

        builder.set_entry_point("retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._route_mode,
            {
                "extractive": "extractive",
                "generative": "generative",
                "__end__":    END,
            },
        )
        builder.add_edge("extractive", END)
        builder.add_edge("generative", END)

        return builder.compile()

    # ── Node methods ──────────────────────────────────────────────────────

    def _node_retrieve(self, state: RAGState) -> dict:
        t0 = time.time()
        documents, num_retrieved = self._retrieve_and_prepare(
            state["question"],
            state["retrieval_n"],
            state["final_n"],
            state["collection"],
            state["retrieval"],
        )
        update: dict = {"documents": documents, "num_retrieved": num_retrieved}
        if state["debug"]:
            update["debug_info"] = {
                "retrieval_time": round(time.time() - t0, 3),
                "initial_count": num_retrieved,
                "final_count": len(documents),
            }
        return update

    def _node_extractive(self, state: RAGState) -> dict:
        t0 = time.time()
        documents = state["documents"]
        candidates = self.extractor.extract(
            state["question"], documents, self.config.max_abstract_length
        )
        answers = [self._build_answer_from_candidate(c, documents) for c in candidates]
        update: dict = {"answers": answers}
        if state["debug"]:
            debug_info = dict(state.get("debug_info") or {})
            debug_info["biobert_scores"] = [
                {
                    "score": float(round(c["score"], 4)),
                    "text": c["text"][:80],
                    "source": getattr(documents[c["doc_idx"]], "source", "?"),
                }
                for c in candidates[:10]
            ]
            debug_info["generation_time"] = round(time.time() - t0, 3)
            update["debug_info"] = debug_info
        return update

    def _node_generative(self, state: RAGState) -> dict:
        t0 = time.time()
        documents = state["documents"]
        messages = self._build_messages(state["question"], documents)
        raw_text = (
            self._generate_vllm(messages)
            if self.config.use_vllm
            else self._generate_cpu(messages)
        )
        answers = self._answers_from_generation(
            state["question"], raw_text, documents, state["mode"]
        )
        update: dict = {"answers": answers}
        if state["debug"]:
            debug_info = dict(state.get("debug_info") or {})
            debug_info["raw_generated_text"] = raw_text
            debug_info["generation_time"] = round(time.time() - t0, 3)
            update["debug_info"] = debug_info
        return update

    @staticmethod
    def _route_mode(state: RAGState) -> str:
        if not state.get("documents"):
            return "__end__"
        return state["mode"]  # "extractive" or "generative"

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        collection: Optional[str] = None,
        return_documents: bool = False,
        debug: bool = False,
        mode: str = "generative",
        retrieval: str = "sparse",
    ) -> Dict:
        """Identical signature to RAGPipeline.run(); orchestrated via LangGraph."""
        start_time = time.time()
        retrieval_n = retrieval_n or self.config.retrieval_n
        final_n = final_n or self.config.final_n

        initial_state: RAGState = {
            "question":         question,
            "retrieval_n":      retrieval_n,
            "final_n":          final_n,
            "collection":       collection,
            "mode":             mode,
            "retrieval":        retrieval,
            "return_documents": return_documents,
            "debug":            debug,
            "start_time":       start_time,
            # populated by nodes:
            "documents":        [],
            "num_retrieved":    0,
            "answers":          [],
            "debug_info":       {} if debug else None,
        }

        final_state = self._graph.invoke(initial_state)

        documents     = final_state["documents"]
        num_retrieved = final_state["num_retrieved"]
        answers       = final_state["answers"]

        response = {
            "sibils_version":          PIPELINE_VERSION,
            "success":                 True,
            "error":                   "",
            "question":                question,
            "collection":              collection or self._default_collection_str,
            "model":                   self._model_label(mode, self.config.model_name),
            "ndocs_requested":         retrieval_n,
            "ndocs_returned_by_SIBiLS": num_retrieved,
            "answers":                 answers,
            "mode_used":               mode,
            "pipeline_time":           round(time.time() - start_time, 3),
            "transformed_query":       None,
        }

        if debug:
            response["debug_info"] = final_state.get("debug_info")

        if return_documents:
            response["documents"] = [
                {
                    "docid":    self._format_docid(doc),
                    "source":   getattr(doc, "source", "faiss"),
                    "title":    doc.title,
                    "abstract": doc.abstract,
                    "pmid":     getattr(doc, "pmid", None),
                    "pmcid":    getattr(doc, "pmcid", None),
                    "doi":      getattr(doc, "doi", None),
                }
                for doc in documents
            ]

        return response
