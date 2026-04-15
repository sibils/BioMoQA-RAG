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
    max_abstract_length: int = 800  # ~200 words — used by BioBERT extractive QA
    llm_abstract_length: int = 600  # shorter cap for LLM prompt to stay within max_model_len=4096
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
                "max_model_len": 4096,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "disable_log_stats": True,
            }

            # Add quantization if specified
            if self.config.quantization:
                vllm_kwargs["quantization"] = self.config.quantization

            self.llm = LLM(**vllm_kwargs)
            print("✓ vLLM model loaded (GPU)")

            # Load tokenizer separately for prompt formatting.
            # llm.chat() + chat_template_kwargs is unreliable across vLLM versions —
            # the enable_thinking kwarg is silently ignored on some builds, causing
            # Qwen3 to generate Chinese thinking tokens instead of answers.
            # Formatting the prompt ourselves via apply_chat_template(enable_thinking=False)
            # is the only approach that works regardless of vLLM version.
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers is required for prompt formatting. Install with: pip install transformers")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            print("✓ Tokenizer loaded for prompt formatting (enable_thinking=False)")

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
        """Return the canonical document identifier based on collection source.

        - medline  → PMID
        - pmc      → PMCID (e.g. PMC7824598)
        - plazi    → Plazi treatment hex ID (the _id from elasticsearch)
        - suppdata → full filename ID (e.g. PMC9700495_zr-43-977-S1.pdf)
        - faiss    → PMID if available, else PMCID
        """
        source = getattr(doc, 'source', None)
        doc_id = getattr(doc, 'doc_id', None)
        if doc_id == 'unknown':
            doc_id = None

        if source == 'pmc':
            pmcid = getattr(doc, 'pmcid', None)
            if pmcid:
                return str(pmcid) if str(pmcid).startswith("PMC") else f"PMC{pmcid}"
            return doc_id  # fallback to _id (e.g. DOI)

        if source in ('plazi', 'suppdata'):
            # doc_id = elasticsearch _id, which is the canonical ID for both
            return doc_id

        # medline / faiss: prefer pmid
        pmid = getattr(doc, 'pmid', None)
        if pmid:
            return str(pmid)
        if doc_id:
            return doc_id
        pmcid = getattr(doc, 'pmcid', None)
        if pmcid:
            return str(pmcid) if str(pmcid).startswith("PMC") else f"PMC{pmcid}"
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

        # Drop documents with no meaningful content (e.g. empty Plazi treatments)
        # Also drop suppdata/pmc docs whose "abstract" is actually CSV/tabular data
        # (bibliography spreadsheets score high on BM25 but BioBERT can't extract from them)
        def _is_prose(doc) -> bool:
            text = ((doc.title or '') + ' ' + (doc.abstract or ''))[:500]
            if len(text.strip()) < 20:
                return False
            # CSV/tabular: high ratio of commas or tabs relative to letters
            letters = sum(1 for c in text if c.isalpha())
            commas = text.count(',') + text.count('\t')
            if letters > 0 and commas / letters > 0.15:
                return False
            return True

        documents = [d for d in documents if _is_prose(d)]

        # Always cap to final_n — ensures BioBERT doesn't run on 30 docs when
        # reranker and relevance filter are both disabled.
        documents = documents[:final_n]

        # Cap per-collection contribution ONLY when mixing multiple collections.
        # When the caller already restricts to a specific collection, all retrieved
        # docs are from that collection and the cap would incorrectly halve them.
        if collection is None:
            from collections import defaultdict
            col_counts: dict = defaultdict(int)
            capped = []
            for d in documents:
                src = getattr(d, 'source', 'unknown')
                limit = final_n // 2 if src != 'medline' else final_n
                if col_counts[src] < limit:
                    capped.append(d)
                    col_counts[src] += 1
            documents = capped

        for d in documents:
            s = float(getattr(d, 'score', 0.0))
            source = getattr(d, 'source', 'faiss')
            if source == 'faiss' or s <= 1.0:
                d.score = round(min(s, 1.0), 4)
            else:
                d.score = round(min(s / 200.0, 1.0), 4)

        return documents, num_retrieved

    @staticmethod
    def _is_garbage(text: str) -> bool:
        """Return True if the generated text looks like degenerate output."""
        import re
        if not text or len(text) < 5:
            return True
        # Any CJK/Hangul/Katakana characters — biomedical answers are in English
        if re.search(r'[\u2e80-\u2eff\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]', text):
            return True
        # More than 10% non-ASCII characters (Cyrillic, Arabic, etc. mixed in)
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.10:
            return True
        # Suspiciously long output (max_tokens=384 ≈ 1 500 chars; > 2 000 is runaway)
        if len(text) > 2000:
            return True
        return False

    def _answers_from_generation(self, question: str, raw_text: str, documents: List, mode: str):
        """Build answers list from a vLLM-generated text string."""
        import re
        # Strip complete <think>...</think> blocks (Qwen3 chain-of-thought)
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # Handle truncated <think> without closing tag (thinking used all max_tokens)
        if '<think>' in text:
            text = text[:text.index('<think>')].strip()
        m = re.compile(
            r"\n+(Okay[,.]|Let me |First[,.]|Moving to|Starting with|Let's tackle)",
            re.IGNORECASE,
        ).search(text)
        if m:
            text = text[:m.start()].strip()
        clean_text = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', text).strip()
        if self._is_garbage(clean_text):
            return []
        # Parse cited [N] indices to surface all referenced documents.
        # If none cited, fall back to top document.
        cited_indices = sorted(set(
            int(n) for n in re.findall(r'\[(\d+)\]', text)
            if int(n) < len(documents)
        ))
        if not cited_indices:
            cited_indices = [0]
        # First item carries the answer text; subsequent items are doc-only (answer="")
        # so the frontend renders the answer once and all cited docs below it.
        return [
            {
                "answer": clean_text if i == 0 else "",
                "answer_score": None,
                "docid": self._format_docid(documents[idx]),
                "doc_source": getattr(documents[idx], 'source', 'faiss'),
                "doc_retrieval_score": round(float(getattr(documents[idx], 'score', 0.0)), 3),
                "doc_text": ((documents[idx].title.strip() + ". " if documents[idx].title and documents[idx].title.strip() else "") + (documents[idx].abstract or ""))[:self.config.max_abstract_length],
                "snippet_start": None,
                "snippet_end": None,
            }
            for i, idx in enumerate(cited_indices)
        ]

    def _build_answer_from_candidate(self, cand: dict, documents: List) -> dict:
        """Build an answer dict from an extractive QA candidate."""
        doc = documents[cand["doc_idx"]]
        # Show a window of ±300 chars around the answer span instead of the full passage.
        passage = cand["passage"]
        s, e = cand["span_start"], cand["span_end"]
        window_start = max(0, s - 500)
        window_end = min(len(passage), e + 500)
        snippet = passage[window_start:window_end].strip()
        return {
            "answer": cand["text"],
            "answer_score": round(cand["score"], 4),
            "docid": self._format_docid(doc),
            "doc_source": getattr(doc, 'source', 'faiss'),
            "doc_retrieval_score": round(float(getattr(doc, 'score', 0.0)), 3),
            "doc_text": snippet,
            "snippet_start": s - window_start,
            "snippet_end": e - window_start,
        }

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

    def run_multi_collection(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        mode: str = "hybrid",
        debug: bool = False,
    ) -> Dict:
        """
        Retrieve independently per collection, then generate answers sequentially.

        Each collection gets its own retrieval pass with final_n document slots so
        no single collection can crowd out others.

        suppdata uses Option B: docs returned directly as answers without QA.
        suppdata retrieval runs in parallel with QA-collection retrieval, and its
        result is built immediately (no LLM wait) so it never blocks QA latency.

        Returns a dict with `collection_results` ordered best-first.
        """
        from concurrent.futures import ThreadPoolExecutor
        start_time = time.time()
        retrieval_n = retrieval_n or self.config.retrieval_n
        final_n = final_n or self.config.final_n

        qa_collections = ["medline", "plazi", "pmc"]
        suppdata_final_n = 3  # suppdata: top 3 docs only (no QA needed)

        # ── 1. All retrievals in parallel (IO-bound: safe to thread) ─────
        # suppdata gets a smaller slot count since we just surface the docs.
        with ThreadPoolExecutor(max_workers=4) as pool:
            qa_futures = {
                col: pool.submit(self._retrieve_and_prepare, question, retrieval_n, final_n, col)
                for col in qa_collections
            }
            supp_future = pool.submit(
                self._retrieve_and_prepare, question, retrieval_n, suppdata_final_n, 'suppdata'
            )
            qa_docs = {col: fut.result() for col, fut in qa_futures.items()}
            supp_docs, supp_n = supp_future.result()

        total_retrieved = sum(n for _, n in qa_docs.values()) + supp_n

        # ── 2. suppdata result — instant, no LLM ─────────────────────────
        suppdata_result = {
            "collection": "suppdata",
            "answers": self._suppdata_doc_answers(supp_docs),
            "mode_used": "document",
        }

        # ── 3. Rank QA collections by their best-scoring document ─────────
        ranked_qa = sorted(
            [(col, docs) for col, (docs, _) in qa_docs.items() if docs],
            key=lambda kv: float(getattr(kv[1][0], 'score', 0.0)) if kv[1] else 0.0,
            reverse=True,
        )

        # ── 4a. BioBERT extraction — parallel across all QA collections ────
        # BioBERT is CPU-bound and independent per collection; run in parallel.
        # Generation must remain sequential (single GPU lock).
        def _run_biobert(col_name, col_docs):
            if mode not in ("extractive", "hybrid"):
                return col_name, [], {}, mode
            candidates = self.extractor.extract(question, col_docs, self.config.max_abstract_length)
            col_debug = {}
            if debug:
                col_debug['biobert_scores'] = [
                    {"score": float(round(c["score"], 4)), "text": c["text"][:80]}
                    for c in candidates[:5]
                ]
            if mode == "hybrid":
                candidates = [c for c in candidates if c["score"] >= self.config.min_extractive_score]
            if candidates:
                mode_used = "extractive" if mode == "extractive" else "hybrid:extractive"
                answers = [self._build_answer_from_candidate(c, col_docs) for c in candidates[:3]]
                return col_name, answers, col_debug, mode_used
            return col_name, [], col_debug, mode

        with ThreadPoolExecutor(max_workers=3) as pool:
            biobert_futures = {
                col: pool.submit(_run_biobert, col, docs)
                for col, docs in ranked_qa
            }
            biobert_results = {col: fut.result() for col, fut in biobert_futures.items()}

        # ── 4b. Generation — sequential (GPU lock) ────────────────────────
        qa_results = []
        final_mode_used = mode

        for col_name, col_docs in ranked_qa:
            _, answers, col_debug, mode_used = biobert_results[col_name]

            if mode == "generative" or (mode == "hybrid" and not answers):
                if mode == "hybrid":
                    mode_used = "hybrid:generative"
                raw_text = (
                    self._generate_vllm(self._build_messages(question, col_docs))
                    if self.config.use_vllm
                    else self._generate_cpu(self._build_messages(question, col_docs))
                )
                if debug:
                    col_debug['raw_generated_text'] = raw_text
                answers = self._answers_from_generation(question, raw_text, col_docs, mode_used)

            if not qa_results:  # first QA collection = best ranked
                final_mode_used = mode_used

            entry = {"collection": col_name, "answers": answers, "mode_used": mode_used}
            if debug:
                entry['debug_info'] = col_debug
            qa_results.append(entry)

        # ── 5. Merge: rank all 4 collections, suppdata by best-doc score ──
        # suppdata rank = based on its best doc score vs QA collections
        supp_score = float(getattr(supp_docs[0], 'score', 0.0)) if supp_docs else 0.0
        all_results = []
        supp_inserted = False
        for entry in qa_results:
            col_name = entry['collection']
            col_docs_list = qa_docs[col_name][0]
            col_score = float(getattr(col_docs_list[0], 'score', 0.0)) if col_docs_list else 0.0
            if not supp_inserted and supp_score >= col_score:
                all_results.append(suppdata_result)
                supp_inserted = True
            all_results.append(entry)
        if not supp_inserted:
            all_results.append(suppdata_result)

        # Add rank numbers
        collection_results = [
            {**entry, "rank": i + 1} for i, entry in enumerate(all_results)
        ]

        result = {
            "question": question,
            "collection_results": collection_results,
            "mode_used": final_mode_used,
            "ndocs_retrieved": total_retrieved,
            "model": self._model_label(final_mode_used, self.config.model_name),
            "pipeline_time": round(time.time() - start_time, 3),
        }
        return result

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

        # Steps 1-4: retrieval → reranking → filtering → score normalization
        t0 = time.time()
        documents, num_retrieved = self._retrieve_and_prepare(
            question, retrieval_n, final_n, collection
        )
        if debug:
            debug_info['retrieval_time'] = round(time.time() - t0, 3)
            debug_info['initial_count'] = num_retrieved
            debug_info['final_count'] = len(documents)

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

        # Option B: suppdata — return documents directly without QA.
        # suppdata files are OCR'd PDFs/spreadsheets with no prose abstracts;
        # BioBERT returns "impossible answer" and the LLM generates garbage.
        if collection == 'suppdata':
            answers = self._suppdata_doc_answers(documents)
            mode_used = "document"

        elif mode in ("extractive", "hybrid"):
            all_candidates = self.extractor.extract(
                question, documents, self.config.max_abstract_length
            )
            if debug:
                debug_info['biobert_scores'] = [
                    {"score": float(round(c["score"], 4)), "text": c["text"][:80], "source": getattr(documents[c["doc_idx"]], "source", "?")}
                    for c in all_candidates[:10]
                ]
            # In hybrid mode, threshold decides generative fallback.
            # In pure extractive, always return best candidates.
            if mode == "hybrid":
                candidates = [c for c in all_candidates if c["score"] >= self.config.min_extractive_score]
            else:
                candidates = all_candidates
            if candidates:
                mode_used = "extractive" if mode == "extractive" else "hybrid:extractive"
                for cand in candidates:
                    answers.append(self._build_answer_from_candidate(cand, documents))

        if collection != 'suppdata' and (mode == "generative" or (mode == "hybrid" and not answers)):
            if mode == "hybrid":
                mode_used = "hybrid:generative"
            raw_text = self._generate_vllm(self._build_messages(question, documents)) \
                if self.config.use_vllm \
                else self._generate_cpu(self._build_messages(question, documents))
            answers = self._answers_from_generation(question, raw_text, documents, mode_used)
            if debug:
                debug_info['raw_generated_text'] = raw_text

        if debug:
            debug_info['generation_time'] = round(time.time() - t0, 3)

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

    @staticmethod
    def _clean_for_llm(text: str) -> str:
        """Strip non-ASCII characters from document text before passing to LLM.

        OCR'd documents (suppdata) and non-English papers often contain garbled
        characters that cause Qwen3 to generate degenerate multilingual output.
        Replacing non-ASCII with a space keeps the prose readable while removing
        the tokens that trigger language-switching in the model.
        """
        import re
        # Replace non-ASCII with space, then collapse runs of whitespace
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()
        return cleaned

    def _suppdata_doc_answers(self, documents: List) -> List[dict]:
        """Option B for suppdata: return retrieved docs directly without running QA.

        suppdata documents are supplementary files (PDFs, spreadsheets) whose
        "abstract" is OCR'd text — BioBERT returns empty spans and the LLM
        produces garbage.  The document title + abstract are returned as-is so
        the frontend can display the full document context.
        """
        answers = []
        for doc in documents[:3]:
            title = (doc.title or '').strip()
            if not title:
                continue
            abstract = (doc.abstract or '').strip()
            # answer = title for the primary display field; doc_text has the full content
            # Include a brief abstract preview in the answer so the frontend has something
            # meaningful to show even if it only renders the answer field.
            answer_text = title
            doc_text = (title + '. ' + abstract)[:self.config.max_abstract_length]
            answers.append({
                "answer": answer_text,
                "answer_score": None,
                "docid": self._format_docid(doc),
                "doc_source": "suppdata",
                "doc_retrieval_score": round(float(getattr(doc, 'score', 0.0)), 3),
                "doc_text": doc_text,
                "snippet_start": None,
                "snippet_end": None,
            })
        return answers

    def _build_messages(self, question: str, documents: List) -> List[dict]:
        """Build chat messages for the LLM from question and retrieved documents.

        Returns a list of message dicts (system + user) for use with llm.chat().
        Uses llm_abstract_length (shorter than BioBERT's max_abstract_length) to
        stay well within vLLM's max_model_len=4096.

        `/no_think` at the end of the user message is Qwen3's soft switch for
        disabling chain-of-thought reasoning at the tokenizer/template level.
        This works regardless of vLLM version, unlike chat_template_kwargs.
        """
        llm_limit = self.config.llm_abstract_length
        context_parts = []
        for i, doc in enumerate(documents):
            abstract = self._clean_for_llm(doc.abstract or '')[:llm_limit]
            title = self._clean_for_llm(doc.title or '')
            docid = self._format_docid(doc) or str(i)
            context_parts.append(f"[{i}] {title}\n{abstract}")

        context = "\n\n".join(context_parts)
        system = (
            "You are a biomedical expert assistant. Answer the question based ONLY "
            "on the provided scientific sources. Be concise and factual (2-4 sentences). "
            "Cite sources using [0], [1], etc. after each claim. "
            "If the sources lack sufficient information, say so briefly. "
            "Always respond in English only."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ]

    def _generate_vllm(self, messages: List[dict]) -> str:
        """Generate with vLLM (GPU).

        Pre-fills the assistant turn with an English phrase so the model
        generates a continuation rather than starting from scratch. This
        bypasses Qwen3 thinking mode (no <think> block fires) and anchors
        the output to English. The prefix is prepended to the stripped
        continuation before returning.
        """
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.15,
            stop=["\nQuestion:", "\nNote:", "\nReferences:", "\nSources:"],
        )

        # Pre-fill the assistant turn so the model continues an English sentence.
        # continue_final_message=True means outputs[0].outputs[0].text contains
        # only the tokens generated AFTER the prefix — we prepend it back.
        ENGLISH_PREFIX = "Based on the provided documents, "
        messages_with_prefix = list(messages) + [
            {"role": "assistant", "content": ENGLISH_PREFIX}
        ]

        with self._generation_lock:
            outputs = self.llm.chat(
                messages_with_prefix,
                sampling_params,
                continue_final_message=True,
                add_generation_prompt=False,
            )
        continuation = outputs[0].outputs[0].text.strip()
        return ENGLISH_PREFIX + continuation

    def _generate_cpu(self, messages: List[dict]) -> str:
        """Generate with transformers (CPU)"""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
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

