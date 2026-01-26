"""
BioMoQA RAG Pipeline for biomedical question answering.

Features:
- Hybrid retrieval (SIBILS + FAISS dense search)
- Cross-encoder reranking
- vLLM generation with sentence-level citations
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import time

# vLLM import (for GPU inference)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Transformers import (for CPU inference)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .retrieval.sibils_retriever import SIBILSRetriever
from .retrieval.dense_retriever import DenseRetriever
from .retrieval.parallel_hybrid import ParallelHybridRetriever, SmartHybridRetriever
from .retrieval.reranker import SemanticReranker
from .retrieval.relevance_filter import FastRelevanceFilter


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline"""
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
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Default to 3B (~6GB VRAM)
    use_vllm: bool = True
    use_cpu: bool = False  # Use CPU inference with transformers instead of vLLM
    quantization: Optional[str] = "fp8"  # FP8 quantization (GPU only)
    gpu_memory_utilization: float = 0.8
    max_tokens: int = 384  # Reduced from 512
    temperature: float = 0.1

    # Context optimization
    max_abstract_length: int = 800  # ~200 words
    truncate_abstracts: bool = True

    # Performance
    enable_parallel: bool = True
    timeout: float = 10.0

    @classmethod
    def cpu_config(cls, model_size: str = "3b") -> "RAGConfig":
        """Create a CPU-optimized configuration with smaller models.

        Args:
            model_size: "1.5b", "3b", or "7b" for Qwen2.5 model sizes
        """
        model_map = {
            "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
            "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
            "3b": "Qwen/Qwen2.5-3B-Instruct",
            "7b": "Qwen/Qwen2.5-7B-Instruct",
        }
        return cls(
            model_name=model_map.get(model_size, model_map["3b"]),
            use_vllm=False,
            use_cpu=True,
            quantization=None,
            max_tokens=256,  # Reduced for CPU
            final_n=5,  # Fewer docs for faster processing
        )

    @classmethod
    def gpu_small_config(cls) -> "RAGConfig":
        """Create a GPU configuration with smaller model (requires ~8GB VRAM)."""
        return cls(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            use_vllm=True,
            use_cpu=False,
            quantization="fp8",
            gpu_memory_utilization=0.8,
            max_tokens=384,
        )


class RAGPipeline:
    """
    BioMoQA RAG Pipeline for biomedical question answering.

    Features:
    - Hybrid retrieval combining SIBILS BM25 and FAISS dense search
    - Cross-encoder reranking for improved relevance
    - vLLM generation with sentence-level citations
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

        print("="*80)
        print("Initializing BioMoQA RAG Pipeline")
        print("="*80)
        print()
        print("Configuration:")
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

        # Load LLM (vLLM for GPU or transformers for CPU)
        self.llm = None
        self.tokenizer = None

        if self.config.use_cpu:
            # CPU inference with transformers
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers is required for CPU inference. Install with: pip install transformers torch")

            print(f"\nLoading model for CPU: {self.config.model_name}")
            print("  Mode: CPU inference (transformers)")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Full precision for CPU
                device_map="cpu"
            )
            print("✓ Transformers model loaded (CPU)")

        elif self.config.use_vllm:
            # GPU inference with vLLM
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is required for GPU inference. Install with: pip install vllm")

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
            print("✓ vLLM model loaded (GPU)")

        print("\n" + "="*80)
        print("BioMoQA RAG Pipeline Ready")
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
        Run the RAG pipeline.

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

        # Step 4: Handle case where no relevant documents found
        if not documents:
            return {
                'question': question,
                'answer': [{
                    'text': 'No relevant biomedical sources were found for this question.',
                    'citation_ids': [],
                    'citations': []
                }],
                'references': [],
                'response_length': 0,
                'pipeline_time': time.time() - start_time,
                'num_retrieved': 0,
                'pipeline_version': '1.0',
            }

        # Step 5: Generate answer with optimizations
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
            'pipeline_version': '1.0',
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

        # Biomedical QA prompt with clear instructions
        prompt = f"""You are a biomedical expert assistant. Answer the question based ONLY on the provided scientific sources.

Instructions:
- Be concise and factual (2-4 sentences unless more detail is needed)
- Cite sources using [0], [1], etc. after each claim
- If the sources don't contain enough information, say "Based on the available sources, this question cannot be fully answered"
- Do not add information not present in the sources

Sources:
{context}

Question: {question}

Answer:"""

        # Generate based on backend
        if self.config.use_cpu:
            answer_text = self._generate_cpu(prompt)
        else:
            answer_text = self._generate_vllm(prompt)

        # Parse answer into sentences with citations
        parsed_answer = self._parse_answer_with_citations(answer_text, documents)

        return parsed_answer

    def _generate_vllm(self, prompt: str) -> str:
        """Generate with vLLM (GPU)"""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _generate_cpu(self, prompt: str) -> str:
        """Generate with transformers (CPU)"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                top_p=0.9 if self.config.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (exclude input)
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _parse_answer_with_citations(self, answer_text: str, documents: List) -> Dict:
        """Parse answer text into sentences with citation extraction"""
        import re

        # Smart sentence splitting that handles abbreviations and numbers
        # First, protect common patterns that shouldn't trigger splits
        protected = answer_text
        protections = [
            (r'(e\.g\.)', '__EG__'),
            (r'(i\.e\.)', '__IE__'),
            (r'(et al\.)', '__ETAL__'),
            (r'(vs\.)', '__VS__'),
            (r'(Dr\.)', '__DR__'),
            (r'(Fig\.)', '__FIG__'),
            (r'(No\.)', '__NO__'),
            (r'(\d+\.\d+)', '__NUM__'),  # Decimal numbers
        ]
        for pattern, placeholder in protections:
            protected = re.sub(pattern, placeholder, protected)

        # Split on period followed by space and capital letter, or end of string
        sentences = re.split(r'\.(?:\s+(?=[A-Z])|\s*$)', protected)

        # Restore protected patterns
        restored_sentences = []
        for sent in sentences:
            restored = sent
            for pattern, placeholder in protections:
                original = pattern.replace('(', '').replace(')', '').replace('\\', '')
                restored = restored.replace(placeholder, original)
            restored_sentences.append(restored)
        sentences = restored_sentences

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
