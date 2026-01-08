"""
V3.1 Pipeline: Ultra-fast inference with quantization and optimizations.

Key optimizations over V3:
1. FP8 quantization (30-40% faster generation)
2. Reduced tokens (384 vs 512)
3. Truncated context (10 docs vs 15)
4. Optimized prompt

Expected: 4.5-5.0s total time (34% faster than V3)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import time
from vllm import LLM, SamplingParams

from src.retrieval.sibils_retriever import SIBILSRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.parallel_hybrid import ParallelHybridRetriever, SmartHybridRetriever
from src.retrieval.reranker import SemanticReranker
from src.retrieval.relevance_filter import FastRelevanceFilter


@dataclass
class RAGConfigV3Fast:
    """V3.1 configuration optimized for maximum speed"""
    # Retrieval
    retrieval_n: int = 20
    use_smart_retrieval: bool = True
    hybrid_alpha: float = 0.5

    # Processing (streamlined)
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_n: int = 15
    use_relevance_filter: bool = True
    final_n: int = 10  # Reduced from 15

    # Generation (optimized)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_vllm: bool = True
    quantization: Optional[str] = "fp8"  # FP8 quantization
    gpu_memory_utilization: float = 0.8
    max_tokens: int = 384  # Reduced from 512
    temperature: float = 0.1

    # Context optimization
    max_abstract_length: int = 800  # ~200 words
    truncate_abstracts: bool = True

    # Performance
    enable_parallel: bool = True
    timeout: float = 10.0


class UltraFastRAGPipeline:
    """
    V3.1 Pipeline optimized for maximum speed.

    Improvements over V3:
    - FP8 quantization: 30-40% faster generation
    - 10 docs instead of 15: Faster processing
    - 384 tokens instead of 512: Faster generation
    - Truncated abstracts: Shorter prompts
    """

    def __init__(self, config: RAGConfigV3Fast = None):
        self.config = config or RAGConfigV3Fast()

        print("="*80)
        print("Initializing BioMoQA RAG Pipeline V3.1 (Ultra-Fast)")
        print("="*80)
        print()
        print("Optimizations:")
        if self.config.quantization:
            print(f"  ✓ Quantization: {self.config.quantization}")
        print(f"  ✓ Max tokens: {self.config.max_tokens}")
        print(f"  ✓ Context docs: {self.config.final_n}")
        print(f"  ✓ Truncate abstracts: {self.config.truncate_abstracts}")
        print()

        # Load retrievers
        print("Loading retrievers...")
        self.sibils = SIBILSRetriever()

        self.dense = DenseRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.dense.load("data/faiss_index.bin", "data/documents.pkl")

        # Smart hybrid retriever
        if self.config.use_smart_retrieval:
            self.retriever = SmartHybridRetriever(
                self.sibils,
                self.dense,
                alpha=self.config.hybrid_alpha,
                k=60
            )
            print("✓ Using SmartHybridRetriever")
        else:
            self.retriever = ParallelHybridRetriever(
                self.sibils,
                self.dense,
                alpha=self.config.hybrid_alpha,
                k=60,
                timeout=self.config.timeout
            )
            print("✓ Using ParallelHybridRetriever")

        # Reranker
        self.reranker = None
        if self.config.use_reranking:
            self.reranker = SemanticReranker(
                model_name=self.config.reranker_model
            )
            print(f"✓ Loaded reranker")

        # Relevance filter
        self.relevance_filter = None
        if self.config.use_relevance_filter:
            self.relevance_filter = FastRelevanceFilter(min_overlap=0.15)
            print("✓ Loaded relevance filter")

        # Load vLLM with quantization
        if self.config.use_vllm:
            print(f"\nLoading vLLM model: {self.config.model_name}")
            if self.config.quantization:
                print(f"  Quantization: {self.config.quantization}")

            vllm_kwargs = {
                "model": self.config.model_name,
                "trust_remote_code": True,
                "max_model_len": 8192,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "disable_log_stats": True,
            }

            # Add quantization if specified
            if self.config.quantization:
                vllm_kwargs["quantization"] = self.config.quantization

            self.llm = LLM(**vllm_kwargs)
            print("✓ vLLM model loaded")

        print("\n" + "="*80)
        print("V3.1 Pipeline Ready (Ultra-Fast Mode)")
        print("="*80)
        print()

    def run(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        return_documents: bool = False,
        debug: bool = False
    ) -> Dict:
        """
        Run V3.1 ultra-fast pipeline.

        Args:
            question: User question
            retrieval_n: Override retrieval count
            final_n: Override final document count
            return_documents: Include documents in response
            debug: Include debug information

        Returns:
            Response dict with answer and metadata
        """
        start_time = time.time()
        retrieval_n = retrieval_n or self.config.retrieval_n
        final_n = final_n or self.config.final_n

        debug_info = {} if debug else None

        # Step 1: Smart retrieval
        t0 = time.time()
        documents = self.retriever.retrieve(
            question,
            n=retrieval_n,
            top_k=retrieval_n
        )
        retrieval_time = time.time() - t0

        if debug:
            debug_info['retrieval_time'] = retrieval_time
            debug_info['initial_count'] = len(documents)

        # Step 2: Optional reranking
        if self.reranker and len(documents) > final_n:
            t0 = time.time()
            documents = self.reranker.rerank(
                question,
                documents,
                top_k=min(self.config.rerank_n, len(documents))
            )
            if debug:
                debug_info['rerank_time'] = time.time() - t0
                debug_info['reranked_count'] = len(documents)

        # Step 3: Fast relevance filtering
        if self.relevance_filter and len(documents) > final_n:
            t0 = time.time()
            documents = self.relevance_filter.filter_relevant(
                question,
                documents,
                max_docs=final_n
            )
            if debug:
                debug_info['filter_time'] = time.time() - t0
                debug_info['filtered_count'] = len(documents)

        # Step 4: Generate answer with optimizations
        t0 = time.time()
        parsed_answer = self.generate_fast(question, documents)
        generation_time = time.time() - t0

        if debug:
            debug_info['generation_time'] = generation_time
            debug_info['final_count'] = len(documents)

        pipeline_time = time.time() - start_time

        # Calculate response length
        response_length = sum(len(sent['text']) for sent in parsed_answer['answer'])

        # Build response (Ragnarok-style format)
        response = {
            'question': question,
            'answer': parsed_answer['answer'],  # List of sentences with citations
            'references': parsed_answer['references'],  # List of cited documents
            'response_length': response_length,
            'pipeline_time': pipeline_time,
            'num_retrieved': len(documents),
            'pipeline_version': 'v3.1-fast',
        }

        if debug:
            response['debug_info'] = debug_info

        if return_documents:
            response['documents'] = [
                {
                    'pmcid': doc.pmcid,
                    'title': doc.title,
                    'abstract': doc.abstract
                } for doc in documents
            ]

        return response

    def generate_fast(self, question: str, documents: List) -> Dict:
        """Generate answer with fast optimizations and parse citations"""

        # Build optimized context
        context_parts = []
        for i, doc in enumerate(documents):
            # Truncate abstract if configured
            abstract = doc.abstract
            if self.config.truncate_abstracts:
                abstract = abstract[:self.config.max_abstract_length]
                if len(doc.abstract) > self.config.max_abstract_length:
                    abstract += "..."

            context_parts.append(
                f"[{i}] PMC{doc.pmcid}: {doc.title}\n{abstract}"
            )

        context = "\n\n".join(context_parts)

        # Optimized prompt (shorter)
        prompt = f"""Answer using context. Cite sources with [0], [1], etc. at the end of each sentence.

Context:
{context}

Question: {question}

Answer:"""

        # Generate with vLLM (fast settings)
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9
        )

        outputs = self.llm.generate([prompt], sampling_params)
        answer_text = outputs[0].outputs[0].text.strip()

        # Parse answer into sentences with citations
        parsed_answer = self._parse_answer_with_citations(answer_text, documents)

        return parsed_answer

    def _parse_answer_with_citations(self, answer_text: str, documents: List) -> Dict:
        """Parse answer text into sentences with citation extraction"""
        import re

        # Split into sentences (simple split on period + space)
        sentences = re.split(r'\.\s+', answer_text)

        parsed_sentences = []
        all_citation_ids = set()

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Find citation markers like [0], [1], [0, 1], etc.
            citation_pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
            citations_found = re.findall(citation_pattern, sentence)

            # Extract unique citation IDs
            citation_ids = []
            for cite_group in citations_found:
                ids = [int(x.strip()) for x in cite_group.split(',')]
                citation_ids.extend(ids)

            # Remove duplicates and sort
            citation_ids = sorted(list(set(citation_ids)))
            all_citation_ids.update(citation_ids)

            # Build citation details
            citation_details = []
            for cid in citation_ids:
                if cid < len(documents):
                    doc = documents[cid]
                    citation_details.append({
                        'document_id': cid,
                        'document_title': doc.title,
                        'pmcid': f"PMC{doc.pmcid}"
                    })

            # Remove citation markers from text for clean display
            clean_text = re.sub(citation_pattern, '', sentence).strip()

            # Add period back if not present
            if clean_text and not clean_text.endswith('.'):
                clean_text += '.'

            parsed_sentences.append({
                'text': clean_text,
                'citation_ids': citation_ids,
                'citations': citation_details
            })

        # Build references list
        references = []
        for i, doc in enumerate(documents):
            if i in all_citation_ids:
                references.append(f"[{i}] PMC{doc.pmcid}: {doc.title}")

        return {
            'answer': parsed_sentences,
            'references': references
        }
