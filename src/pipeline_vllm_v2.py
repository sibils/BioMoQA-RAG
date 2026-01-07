"""
Enhanced RAG Pipeline V2 with Improved Retrieval

Improvements over V1:
1. ✅ Semantic reranking (cross-encoder)
2. ✅ Query expansion (LLM + rules)
3. ✅ Relevance filtering (keyword + LLM)
4. ✅ Better context selection

Expected: +40-60% improvement in answer quality
"""

from typing import List, Dict, Optional
import time
from dataclasses import dataclass
from vllm import LLM, SamplingParams

from .retrieval import SIBILSRetriever, Document
from .retrieval.reranker import SemanticReranker, HybridReranker
from .retrieval.query_expander import HybridQueryExpander
from .retrieval.relevance_filter import FastRelevanceFilter, HybridRelevanceFilter
from .generation import LLMGenerator


@dataclass
class RAGConfigV2:
    """Enhanced configuration for V2 pipeline."""
    # Retrieval
    retrieval_n: int = 100  # Retrieve more initially
    retrieval_collection: str = "pmc"

    # Query expansion
    use_query_expansion: bool = True
    n_query_variants: int = 1  # Number of LLM-generated variants

    # Reranking
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_n: int = 30  # Rerank top-30 to get best 20

    # Relevance filtering
    use_relevance_filter: bool = True
    relevance_filter_type: str = "fast"  # "fast", "llm", or "hybrid"
    final_n: int = 20  # Final number of docs for generation

    # Generation
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_vllm: bool = True
    gpu_memory_utilization: float = 0.8
    max_new_tokens: int = 512
    temperature: float = 0.7

    # General
    device: str = "cuda"


class EnhancedRAGPipeline:
    """
    V2 RAG Pipeline with improved retrieval quality.

    Performance targets:
    - Speed: ~8-12 seconds (slightly slower than V1 due to reranking)
    - Quality: +40-60% better than V1 (60-70% ROUGE-1 expected)
    - Citation quality: More relevant citations
    """

    def __init__(self, config: Optional[RAGConfigV2] = None):
        """Initialize enhanced pipeline."""
        self.config = config or RAGConfigV2()

        print("="*80)
        print("Initializing Enhanced BioMoQA RAG Pipeline V2")
        print("="*80)
        print(f"\nImprovements:")
        print(f"  ✓ Query expansion: {self.config.use_query_expansion}")
        print(f"  ✓ Semantic reranking: {self.config.use_reranking}")
        print(f"  ✓ Relevance filtering: {self.config.use_relevance_filter}")
        print()

        # Initialize retriever
        print("Loading retriever...")
        self.retriever = SIBILSRetriever(
            collection=self.config.retrieval_collection,
            default_n=self.config.retrieval_n,
        )

        # Initialize vLLM
        if self.config.use_vllm:
            print(f"Loading vLLM model: {self.config.model_name}")
            print("This model will be used for:")
            print("  - Query expansion")
            print("  - Answer generation")
            if self.config.relevance_filter_type in ["llm", "hybrid"]:
                print("  - Relevance filtering")

            self.llm = LLM(
                model=self.config.model_name,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=8192,
            )

            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=0.9,
            )

            print("✓ vLLM loaded successfully!\n")
        else:
            print("Loading standard generator (slower)...")
            self.generator = LLMGenerator(
                model_name=self.config.model_name,
                device=self.config.device,
                load_in_4bit=True,
                max_new_tokens=self.config.max_new_tokens,
            )
            self.llm = None

        # Initialize query expander
        if self.config.use_query_expansion:
            print("Loading query expander...")
            self.query_expander = HybridQueryExpander(
                llm=self.llm,
                sampling_params=self.sampling_params
            )
            print("✓ Query expander ready\n")
        else:
            self.query_expander = None

        # Initialize reranker
        if self.config.use_reranking:
            print(f"Loading reranker: {self.config.reranker_model}")
            self.reranker = SemanticReranker(
                model_name=self.config.reranker_model
            )
            print()
        else:
            self.reranker = None

        # Initialize relevance filter
        if self.config.use_relevance_filter:
            print(f"Loading relevance filter: {self.config.relevance_filter_type}")
            if self.config.relevance_filter_type == "fast":
                self.relevance_filter = FastRelevanceFilter(min_overlap=0.15)
            elif self.config.relevance_filter_type == "llm":
                from .retrieval.relevance_filter import LLMRelevanceFilter
                self.relevance_filter = LLMRelevanceFilter(
                    llm=self.llm,
                    min_relevant=self.config.final_n // 2
                )
            else:  # hybrid
                self.relevance_filter = HybridRelevanceFilter(
                    llm=self.llm,
                    min_overlap=0.15,
                    min_relevant=self.config.final_n // 2
                )
            print("✓ Relevance filter ready\n")
        else:
            self.relevance_filter = None

        print("="*80)
        print("V2 Pipeline Ready!")
        print("="*80)
        print()

    def create_rag_prompt(self, question: str, documents: List[Document]) -> str:
        """Create RAG prompt (same as V1)."""
        context_str = ""
        for i, doc in enumerate(documents):
            title = doc.title
            text = doc.get_text(max_length=1000)
            context_str += f"[{i}] {title}\n{text}\n\n"

        prompt = f"""System: Answer the question using the provided context documents. Cite sources using [0], [1], etc.

QUESTION: {question}

CONTEXTS:
{context_str}

ANSWER:"""

        return prompt

    def parse_citations(self, answer_text: str, documents: List[Document]) -> List[Dict]:
        """Parse answer into sentences with explicit citations (same as V1)."""
        import re

        sentences = []
        raw_sentences = re.split(r'(?<=[.!?])\s+', answer_text)

        for sent in raw_sentences:
            if not sent.strip():
                continue

            citation_pattern = r'\[(\d+)\]'
            matches = re.findall(citation_pattern, sent)
            citation_ids = [int(m) for m in matches]

            # Build explicit citation details
            citation_details = []
            for cid in citation_ids:
                if cid < len(documents):
                    doc = documents[cid]
                    citation_details.append({
                        "document_id": cid,
                        "document_title": doc.title,
                        "pmcid": doc.doc_id,
                    })

            clean_text = re.sub(citation_pattern, '', sent).strip()

            if clean_text:
                sentences.append({
                    "text": clean_text,
                    "citation_ids": citation_ids,
                    "citations": citation_details,
                })

        return sentences

    def generate_vllm(self, question: str, documents: List[Document]) -> str:
        """Generate answer using vLLM."""
        prompt = self.create_rag_prompt(question, documents)
        outputs = self.llm.generate([prompt], self.sampling_params)
        answer_text = outputs[0].outputs[0].text.strip()
        return answer_text

    def run(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        rerank_n: Optional[int] = None,
        final_n: Optional[int] = None,
        topic_id: Optional[str] = None,
        return_documents: bool = False,
        debug: bool = False,
    ) -> Dict:
        """
        Run enhanced RAG pipeline V2.

        Args:
            question: Input question
            retrieval_n: Number of docs to retrieve initially
            rerank_n: Number of docs after reranking
            final_n: Number of docs for generation
            topic_id: Optional topic ID
            return_documents: Include retrieved docs in output
            debug: Print debug info about retrieval steps

        Returns:
            Dict with answer, citations, and metadata
        """
        start_time = time.time()
        debug_info = {}

        retrieval_n = retrieval_n or self.config.retrieval_n
        rerank_n = rerank_n or self.config.rerank_n
        final_n = final_n or self.config.final_n

        # Step 1: Query Expansion
        if self.config.use_query_expansion and self.query_expander:
            expanded = self.query_expander.expand(question, n_llm_variants=self.config.n_query_variants)
            queries = expanded.all_queries
            if debug:
                print(f"Query expansion: {len(queries)} queries")
                for i, q in enumerate(queries):
                    print(f"  {i+1}. {q}")
            debug_info["expanded_queries"] = queries
        else:
            queries = [question]

        # Step 2: Multi-query Retrieval
        all_documents = []
        seen_ids = set()

        for query in queries:
            docs = self.retriever.retrieve(query, n=retrieval_n // len(queries) + 10)
            for doc in docs:
                if doc.doc_id not in seen_ids:
                    all_documents.append(doc)
                    seen_ids.add(doc.doc_id)

        # Take top retrieval_n
        all_documents = all_documents[:retrieval_n]

        if debug:
            print(f"\nRetrieved {len(all_documents)} documents")

        debug_info["initial_retrieval_count"] = len(all_documents)

        # Step 3: Reranking
        if self.config.use_reranking and self.reranker and len(all_documents) > 0:
            all_documents = self.reranker.rerank(question, all_documents, top_k=rerank_n)
            if debug:
                print(f"After reranking: {len(all_documents)} documents")
            debug_info["reranked_count"] = len(all_documents)

        # Step 4: Relevance Filtering
        if self.config.use_relevance_filter and self.relevance_filter and len(all_documents) > 0:
            all_documents = self.relevance_filter.filter_relevant(question, all_documents, max_docs=final_n)
            if debug:
                print(f"After filtering: {len(all_documents)} documents")
            debug_info["filtered_count"] = len(all_documents)
        else:
            all_documents = all_documents[:final_n]

        # Final document set
        documents = all_documents

        if debug:
            print(f"\nFinal document set: {len(documents)} documents")
            print()

        debug_info["final_count"] = len(documents)

        # Step 5: Generation with vLLM
        if self.config.use_vllm:
            answer_text = self.generate_vllm(question, documents)
        else:
            result = self.generator.generate(
                question,
                [{"title": d.title, "text": d.get_text(1000)} for d in documents]
            )
            answer_text = result["answer_text"]

        # Parse citations with explicit details
        answer_sentences = self.parse_citations(answer_text, documents)

        # Build references list with full document info
        references = []
        for i, doc in enumerate(documents):
            references.append(f"[{i}] {doc.doc_id}: {doc.title}")

        # Build output
        output = {
            "topic_id": topic_id or "unknown",
            "question": question,
            "references": references,
            "response_length": len(answer_text),
            "answer": answer_sentences,
            "raw_answer": answer_text,
            "pipeline_time": round(time.time() - start_time, 2),
            "num_retrieved": len(documents),
            "pipeline_version": "v2",
        }

        if debug:
            output["debug_info"] = debug_info

        if return_documents:
            output["documents"] = [
                {
                    "id": doc.doc_id,
                    "title": doc.title,
                    "score": doc.score,
                    "abstract": doc.abstract[:200],
                }
                for doc in documents
            ]

        return output
