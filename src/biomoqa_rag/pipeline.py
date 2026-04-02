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

PIPELINE_VERSION = "biomoqa-2.0"

from .retrieval.sibils_retriever import SIBILSRetriever
from .retrieval.dense_retriever import DenseRetriever
from .retrieval.parallel_hybrid import ParallelHybridRetriever, SmartHybridRetriever
from .retrieval.reranker import SemanticReranker
from .retrieval.relevance_filter import FastRelevanceFilter


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline"""
    # Retrieval
    retrieval_n: int = 30
    use_smart_retrieval: bool = True
    hybrid_alpha: float = 0.5
    sibils_collections: List = None  # None = SIBILS default (medline + plazi)

    # Processing (streamlined)
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_n: int = 15
    use_relevance_filter: bool = True
    final_n: int = 5
    min_extractive_score: float = 0.30  # fallback to generative if BioBERT is less confident

    # Generation (optimized)
    model_name: str = "Qwen/Qwen3-8B"  # Qwen3-8B (~8GB VRAM with fp8)
    use_vllm: bool = True
    use_cpu: bool = False  # Use CPU inference with transformers instead of vLLM
    quantization: Optional[str] = "fp8"  # FP8 quantization (GPU only)
    gpu_memory_utilization: float = 0.83
    max_tokens: int = 384
    temperature: float = 0.1

    # Context optimization
    max_abstract_length: int = 800  # ~200 words
    truncate_abstracts: bool = True

    # Extractive QA (optional mode — lazy-loaded on first use)
    qa_model: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
    qa_confidence_threshold: float = 0.01  # Lowered from 0.1 — BioBERT is conservative
    qa_device: int = 0  # 0 = GPU, -1 = CPU

    # SIBILS disk cache
    sibils_cache_dir: Optional[str] = "data/sibils_cache"
    sibils_cache_ttl: int = 604800  # 7 days

    # Performance
    enable_parallel: bool = True
    timeout: float = 30.0

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
        """Create a GPU configuration with a smaller model (requires ~4GB VRAM)."""
        return cls(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            use_vllm=True,
            use_cpu=False,
            quantization="fp8",
            gpu_memory_utilization=0.83,
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
        self._generation_lock = __import__("threading").Lock()

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
        self.sibils = SIBILSRetriever(
            collection=self.config.sibils_collections,
            cache_dir=self.config.sibils_cache_dir,
            cache_ttl=self.config.sibils_cache_ttl,
        )

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

        # Extractive QA model — lazy-loaded on first use
        self._extractor = None

        print("\n" + "="*80)
        print("BioMoQA RAG Pipeline Ready")
        print("="*80)
        print()

    @property
    def extractor(self):
        """Lazy-load the BioBERT extractive QA model on first use."""
        if self._extractor is None:
            from .extraction.extractive_qa import BioExtractiveQA
            print(f"Loading BioBERT QA model: {self.config.qa_model}")
            self._extractor = BioExtractiveQA(
                model_name=self.config.qa_model,
                confidence_threshold=self.config.qa_confidence_threshold,
                device=self.config.qa_device,
            )
            print("✓ BioBERT QA model loaded")
        return self._extractor

    @staticmethod
    def _format_docid(doc) -> Optional[str]:
        """Return the canonical document identifier based on collection source."""
        source = getattr(doc, 'source', None)
        if source == 'pmc':
            pmcid = getattr(doc, 'pmcid', None)
            if pmcid:
                return pmcid if str(pmcid).startswith("PMC") else f"PMC{pmcid}"
        if source == 'suppdata':
            doc_id = getattr(doc, 'doc_id', None)
            if doc_id and doc_id != 'unknown':
                return str(doc_id)
        # medline, plazi, faiss: prefer pmid, fallback to doc_id, then pmcid
        if getattr(doc, 'pmid', None):
            return str(doc.pmid)
        if getattr(doc, 'doc_id', None) and doc.doc_id != 'unknown':
            return str(doc.doc_id)
        if getattr(doc, 'pmcid', None):
            pmcid = doc.pmcid
            return pmcid if str(pmcid).startswith("PMC") else f"PMC{pmcid}"
        return None

    @staticmethod
    def _model_label(mode_used: str, model_name: str) -> str:
        """Short model name for the response 'model' field."""
        if "extractive" in mode_used:
            return "biobert"
        return model_name.split("/")[-1].lower()

    def _retrieve_and_prepare(
        self,
        question: str,
        retrieval_n: int,
        final_n: int,
        collection: Optional[str],
    ):
        """Steps 1-4: retrieval → reranking → filtering → score normalization.

        Thread-safe (no GPU, only SIBILS HTTP + FAISS + CPU CrossEncoder).
        Returns (documents, num_retrieved).
        """
        documents = self.retriever.retrieve(
            question, n=retrieval_n, top_k=retrieval_n, collection=collection,
        )
        num_retrieved = len(documents)

        if self.reranker and len(documents) > final_n:
            documents = self.reranker.rerank(
                question, documents, top_k=min(self.config.rerank_n, len(documents))
            )

        if self.relevance_filter and len(documents) > final_n:
            documents = self.relevance_filter.filter_relevant(
                question, documents, max_docs=final_n
            )

        for d in documents:
            s = float(getattr(d, 'score', 0.0))
            source = getattr(d, 'source', 'faiss')
            if source == 'faiss' or s <= 1.0:
                d.score = round(min(s, 1.0), 4)
            else:
                d.score = round(min(s / 200.0, 1.0), 4)

        return documents, num_retrieved

    def _answers_from_generation(self, question: str, raw_text: str, documents: List, mode: str):
        """Build answers list from a vLLM-generated text string."""
        import re
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        m = re.compile(
            r"\n+(Okay[,.]|Let me |First[,.]|Moving to|Starting with|Let's tackle)",
            re.IGNORECASE,
        ).search(text)
        if m:
            text = text[:m.start()].strip()
        clean_text = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', text).strip()
        top_doc = documents[0]
        return [{
            "answer": clean_text,
            "answer_score": None,
            "docid": self._format_docid(top_doc),
            "doc_source": getattr(top_doc, 'source', 'faiss'),
            "doc_retrieval_score": round(float(getattr(top_doc, 'score', 0.0)), 3),
            "doc_text": f"{top_doc.title}. {top_doc.abstract}"[:self.config.max_abstract_length],
            "snippet_start": None,
            "snippet_end": None,
        }]

    @property
    def _default_collection_str(self) -> str:
        c = self.sibils.collection
        return '+'.join(c) if isinstance(c, list) else (c or 'medline+plazi')

    def _build_response(
        self,
        question: str,
        answers: List,
        documents: List,
        num_retrieved: int,
        retrieval_n: int,
        collection: Optional[str],
        mode_used: str,
        pipeline_time: float,
    ) -> Dict:
        return {
            'sibils_version': PIPELINE_VERSION,
            'success': True,
            'error': '',
            'question': question,
            'collection': collection or self._default_collection_str,
            'model': self._model_label(mode_used, self.config.model_name),
            'ndocs_requested': retrieval_n,
            'ndocs_returned_by_SIBiLS': num_retrieved,
            'answers': answers,
            'mode_used': mode_used,
            'pipeline_time': round(pipeline_time, 3),
            'transformed_query': None,
        }

    def run_batch(
        self,
        questions: List[str],
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> List[Dict]:
        """Run extractive QA on multiple questions with parallel retrieval.

        Retrieval (SIBILS HTTP + FAISS + reranking) runs concurrently across
        all questions — I/O-bound, thread-safe.  Results are returned in the
        same order as the input list.
        """
        from concurrent.futures import ThreadPoolExecutor
        start_time = time.time()
        retrieval_n = retrieval_n or self.config.retrieval_n
        final_n = final_n or self.config.final_n

        # ── 1. Parallel retrieval ──────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=min(len(questions), 8)) as pool:
            retrieval_results = list(pool.map(
                lambda q: self._retrieve_and_prepare(q, retrieval_n, final_n, collection),
                questions,
            ))

        # ── 2. Extractive pass for each question ───────────────────────────
        responses = []
        for question, (documents, num_retrieved) in zip(questions, retrieval_results):
            answers = []
            if documents:
                candidates = self.extractor.extract(
                    question, documents, self.config.max_abstract_length
                )
                candidates = [c for c in candidates
                              if c["score"] >= self.config.min_extractive_score]
                for cand in candidates:
                    doc = documents[cand["doc_idx"]]
                    answers.append({
                        "answer": cand["text"],
                        "answer_score": round(cand["score"], 4),
                        "docid": self._format_docid(doc),
                        "doc_source": getattr(doc, 'source', 'faiss'),
                        "doc_retrieval_score": round(float(getattr(doc, 'score', 0.0)), 3),
                        "doc_text": cand["passage"],
                        "snippet_start": cand["span_start"],
                        "snippet_end": cand["span_end"],
                    })
            responses.append(self._build_response(
                question, answers, documents, num_retrieved,
                retrieval_n, collection, "extractive", time.time() - start_time,
            ))

        return responses

    def run(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        collection: Optional[str] = None,
        return_documents: bool = False,
        debug: bool = False,
        mode: str = "hybrid",
    ) -> Dict:
        """
        Run the RAG pipeline.

        Args:
            question: User question
            retrieval_n: Override retrieval count
            final_n: Override final document count
            collection: Override SIBILS collection ("medline", "plazi", etc.).
                        If None, uses default (medline + plazi).
            return_documents: Include documents in response
            debug: Include debug information
            mode: Answer strategy:
                  - "hybrid" (default): extractive first, generative fallback
                  - "extractive": verbatim span from BioBERT, no hallucination possible
                  - "generative": LLM synthesises a multi-sentence answer

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
            top_k=retrieval_n,
            collection=collection,
        )
        num_retrieved = len(documents)  # total before reranking/filtering
        retrieval_time = time.time() - t0

        if debug:
            debug_info['retrieval_time'] = round(retrieval_time, 3)
            debug_info['initial_count'] = num_retrieved

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

        # Normalize retrieval scores to [0, 1].
        # FAISS cosine scores are already in [0, 1].
        # BM25 scores can be >100 — cap at 200 and divide (covers typical range).
        for d in documents:
            s = float(getattr(d, 'score', 0.0))
            source = getattr(d, 'source', 'faiss')
            if source == 'faiss' or s <= 1.0:
                d.score = round(min(s, 1.0), 4)
            else:
                d.score = round(min(s / 200.0, 1.0), 4)

        # Step 4: Handle case where no relevant documents found
        if not documents:
            return {
                'sibils_version': PIPELINE_VERSION,
                'success': True,
                'error': '',
                'question': question,
                'collection': collection or self._default_collection_str,
                'model': self._model_label(mode, self.config.model_name),
                'ndocs_requested': retrieval_n,
                'ndocs_returned_by_SIBiLS': 0,
                'answers': [],
                'mode_used': mode,
                'pipeline_time': round(time.time() - start_time, 3),
                'transformed_query': None,
            }

        # Step 5: Generate or extract answer
        t0 = time.time()
        mode_used = mode
        answers = []

        if mode in ("extractive", "hybrid"):
            candidates = self.extractor.extract(
                question, documents, self.config.max_abstract_length
            )
            candidates = [c for c in candidates if c["score"] >= self.config.min_extractive_score]
            if candidates:
                mode_used = "extractive" if mode == "extractive" else "hybrid:extractive"
                for cand in candidates:
                    doc = documents[cand["doc_idx"]]
                    answers.append({
                        "answer": cand["text"],
                        "answer_score": round(cand["score"], 4),
                        "docid": self._format_docid(doc),
                        "doc_source": getattr(doc, 'source', 'faiss'),
                        "doc_retrieval_score": round(float(getattr(doc, 'score', 0.0)), 3),
                        "doc_text": cand["passage"],
                        "snippet_start": cand["span_start"],
                        "snippet_end": cand["span_end"],
                    })

        if mode == "generative" or (mode == "hybrid" and not answers):
            if mode == "hybrid":
                mode_used = "hybrid:generative"
            raw_text = self._generate_vllm(self._build_prompt(question, documents)) \
                if self.config.use_vllm \
                else self._generate_cpu(self._build_prompt(question, documents))
            import re
            clean_text = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', raw_text).strip()
            top_doc = documents[0]
            answers.append({
                "answer": clean_text,
                "answer_score": None,
                "docid": self._format_docid(top_doc),
                "doc_source": getattr(top_doc, 'source', 'faiss'),
                "doc_retrieval_score": round(float(getattr(top_doc, 'score', 0.0)), 3),
                "doc_text": f"{top_doc.title}. {top_doc.abstract}"[:self.config.max_abstract_length],
                "snippet_start": None,
                "snippet_end": None,
            })

        if debug:
            debug_info['generation_time'] = round(time.time() - t0, 3)
            debug_info['final_count'] = len(documents)

        response = {
            'sibils_version': PIPELINE_VERSION,
            'success': True,
            'error': '',
            'question': question,
            'collection': collection or self._default_collection_str,
            'model': self._model_label(mode_used, self.config.model_name),
            'ndocs_requested': retrieval_n,
            'ndocs_returned_by_SIBiLS': num_retrieved,
            'answers': answers,
            'mode_used': mode_used,
            'pipeline_time': round(time.time() - start_time, 3),
            'transformed_query': None,
        }

        if debug:
            response['debug_info'] = debug_info

        if return_documents:
            response['documents'] = [
                {
                    'docid': self._format_docid(doc),
                    'source': getattr(doc, 'source', 'faiss'),
                    'title': doc.title,
                    'abstract': doc.abstract,
                    'pmid': getattr(doc, 'pmid', None),
                    'pmcid': getattr(doc, 'pmcid', None),
                    'doi': getattr(doc, 'doi', None),
                } for doc in documents
            ]

        return response

    def _build_prompt(self, question: str, documents: List) -> str:
        """Build the LLM prompt from question and retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents):
            abstract = doc.abstract
            if self.config.truncate_abstracts:
                abstract = abstract[:self.config.max_abstract_length]
                if len(doc.abstract) > self.config.max_abstract_length:
                    abstract += "..."
            docid = self._format_docid(doc) or str(i)
            context_parts.append(f"[{i}] {docid}: {doc.title}\n{abstract}")

        context = "\n\n".join(context_parts)
        # Qwen3 chat format with thinking disabled.
        # Pre-filling <think>\n\n</think> tells the model its thinking is done
        # and it should output the answer directly, preventing reasoning leakage.
        system = (
            "You are a biomedical expert assistant. Answer the question based ONLY "
            "on the provided scientific sources. Be concise and factual (2-4 sentences). "
            "Cite sources using [0], [1], etc. after each claim. "
            "If the sources lack sufficient information, say so briefly."
        )
        user = f"Sources:\n{context}\n\nQuestion: {question}"
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n"
        )

    def _generate_vllm(self, prompt: str) -> str:
        """Generate with vLLM (GPU)"""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9,
            repetition_penalty=1.3,
        )

        with self._generation_lock:
            outputs = self.llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text.strip()
        # Qwen3 sometimes emits chain-of-thought reasoning after its answer even
        # with the <think></think> prefix. Strip any <think>...</think> blocks,
        # then cut at common reasoning openers if they appear mid-answer.
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        _reasoning_re = re.compile(
            r"\n+(Okay[,.]|Let me |First[,.]|Moving to|Starting with|Let's tackle)",
            re.IGNORECASE,
        )
        m = _reasoning_re.search(text)
        if m:
            text = text[:m.start()].strip()
        return text

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

