"""
BioMoQA RAG pipeline for biomedical question answering.

For one question and one collection:
1. Search, at the depth asked (depth.py): fast runs two keyword searches and keeps the
   five documents a cross-encoder scores best against the question; deep adds searches
   written by the language model.
2. Read the five: the language model writes a short answer citing them (generative), or
   BioBERT copies the best span (extractive). When the language model is slow or down,
   the extractive answer is sent instead, and the response says so.
"""

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# vLLM import (for in-process GPU inference)
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
from .retrieval.reranker import SemanticReranker
from .passage import best_passage
from .answer_signals import annotate, mean_token_probability, rank_collection_results
from .depth import (
    DEFAULT_DEPTH, DEPTHS, DepthSettings, SubQueryWriter, merge, note, select, sigmoid,
)
from .llm_remote import LLMUnavailable, RemoteLLM

logger = logging.getLogger("biomoqa")

# The assistant turn is pre-filled with this English phrase: the model continues the
# sentence, which anchors the answer in English and keeps it from thinking first.
ANSWER_PREFIX = "Based on the provided documents, "
ANSWER_STOP = ["\nQuestion:", "\nNote:", "\nReferences:", "\nSources:"]


class GeneratedText(str):
    """Generated text that also carries the generation's mean token probability.

    A str, so every caller of _generate_vllm keeps working unchanged; the answer
    builder reads `.confidence` when it is there.
    """
    confidence = None

    def __new__(cls, text: str, confidence=None):
        obj = super().__new__(cls, text)
        obj.confidence = confidence
        return obj


# Habitat questions: the reader is asked for the environment, not the collection localities.
import re as _re
_HABITAT_Q = _re.compile(
    r"\bhabitats?\b|\bwhere (?:does|do|is|are|can)\b.*\b(?:live|found|occur|inhabit)\w*"
    r"|\binhabit\w*\b|\becolog\w*\b", _re.I)

# Citation markers: [N] or [N, M]. A run of them ("[1], [2]") goes with the space before it,
# so the answer keeps no " , ." and no space before its full stop. "[18F]altanserin" is no marker.
_MARKER = r"\[\d+(?:\s*,\s*\d+)*\]"
_MARKER_RUN = _re.compile(rf"\s*{_MARKER}(?:\s*,?\s*{_MARKER})*")


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline"""
    # Retrieval
    retrieval_n: int = 20             # documents per search; the depth pipeline merges the lists
    sibils_collections: List = None   # None = SIBILS default (medline + plazi)
    default_depth: str = DEFAULT_DEPTH
    depth: DepthSettings = field(default_factory=DepthSettings)

    # Reranking (every candidate is scored against the question as asked)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_device: str = "cpu"
    final_n: int = 5
    min_extractive_score: float = 0.30  # run_batch drops extractive answers below this

    # Generation: "remote" = the shared language-model server; "vllm" = a model loaded
    # in this process on the GPU; "cpu" = transformers on the CPU.
    generation_backend: str = "remote"
    llm_base_url: str = "http://llm.lan.text-analytics.ch/qwen/v1"
    llm_model: str = "Qwen3.8-27B"
    llm_first_token_timeout: float = 4.0
    llm_total_timeout: float = 20.0
    llm_priority: int = -10
    # Batch jobs (a POST that does not come from a web browser): low priority on the shared
    # server, long limits, and never BioBERT's answer in place of the generated one.
    llm_batch_priority: int = 10
    llm_batch_first_token_timeout: float = 60.0
    llm_batch_total_timeout: float = 100.0

    # In-process generation (generation_backend "vllm" or "cpu")
    model_name: str = "Qwen/Qwen3-8B"
    use_vllm: bool = True
    use_cpu: bool = False
    quantization: Optional[str] = "fp8"
    gpu_memory_utilization: float = 0.83
    max_model_len: int = 4096
    max_tokens: int = 300
    temperature: float = 0.2

    # Context
    max_abstract_length: int = 800    # BioBERT (the extractive reader reads up to 8,000 characters whole)
    llm_abstract_length: int = 600    # window the language model gets from a long document
    llm_whole_upto: int = 2500        # documents up to this length are given whole
    truncate_abstracts: bool = True

    # Extractive QA (lazy-loaded on first use)
    qa_model: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
    qa_confidence_threshold: float = 0.01
    qa_device: int = 0  # 0 = GPU, -1 = CPU

    # SIBILS disk cache
    sibils_cache_dir: Optional[str] = "data/sibils_cache"
    sibils_cache_ttl: int = 604800  # 7 days
    sibils_empty_cache_ttl: int = 0  # never serve empty results from cache
    sibils_enforce_taxon_phrase: bool = True  # require recognised binomials as a phrase

    timeout: float = 30.0

    def __post_init__(self):
        if self.generation_backend != "vllm":
            self.use_vllm = False
        if self.generation_backend == "cpu":
            self.use_cpu = True

    @property
    def generator_label(self) -> str:
        return self.llm_model if self.generation_backend == "remote" else self.model_name

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
            generation_backend="cpu",
            model_name=model_map.get(model_size, model_map["3b"]),
            quantization=None,
            max_tokens=256,  # Reduced for CPU
            final_n=5,  # Fewer docs for faster processing
        )

    @classmethod
    def gpu_small_config(cls) -> "RAGConfig":
        """Create a GPU configuration with a smaller model (requires ~4GB VRAM)."""
        return cls(
            generation_backend="vllm",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            quantization="fp8",
            gpu_memory_utilization=0.83,
            max_tokens=384,
        )


class RAGPipeline:
    """
    BioMoQA RAG pipeline: depth search, cross-encoder reranking, and a generative
    (language model) or extractive (BioBERT) reader.
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self._generation_lock = __import__("threading").Lock()

        print("="*80)
        print("Initializing BioMoQA RAG Pipeline")
        print("="*80)
        print()
        print("Configuration:")
        print(f"  ✓ Generation: {self.config.generation_backend} ({self.config.generator_label})")
        print(f"  ✓ Default depth: {self.config.default_depth}")
        print(f"  ✓ Max tokens: {self.config.max_tokens}")
        print(f"  ✓ Context docs: {self.config.final_n}")
        print()

        # Load retrievers
        print("Loading retrievers...")
        self.sibils = SIBILSRetriever(
            collection=self.config.sibils_collections,
            cache_dir=self.config.sibils_cache_dir,
            cache_ttl=self.config.sibils_cache_ttl,
            empty_cache_ttl=self.config.sibils_empty_cache_ttl,
            enforce_taxon_phrase=self.config.sibils_enforce_taxon_phrase,
        )
        # Kept for callers of the old attribute names (doc_resolver, eval scripts).
        self.sibils_retriever = self.sibils
        self.retriever = self.sibils
        print("✓ SIBILS retriever")

        self.reranker = SemanticReranker(
            model_name=self.config.reranker_model, device=self.config.reranker_device)
        print("✓ Loaded reranker")

        # Language model
        self.llm = None
        self.tokenizer = None
        self.remote: Optional[RemoteLLM] = None
        self.subqueries: Optional[SubQueryWriter] = None

        if self.config.generation_backend == "remote":
            self.remote = RemoteLLM(
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
                first_token_timeout=self.config.llm_first_token_timeout,
                total_timeout=self.config.llm_total_timeout,
                priority=self.config.llm_priority,
            )
            # The deep mode's searches are written by the same server.
            self.subqueries = SubQueryWriter(self.remote.chat, self.config.depth,
                                             batch_priority=self.config.llm_batch_priority)
            print(f"✓ Generation through the shared server: {self.config.llm_model}")

        elif self.config.use_cpu:
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
                "max_model_len": self.config.max_model_len,
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
            from transformers import AutoTokenizer as _AutoTokenizer
            self.tokenizer = _AutoTokenizer.from_pretrained(
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

        if source == 'plazi':
            return doc_id

        # medline: prefer pmid
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

    # ------------------------------------------------------------------
    # Search at the depth asked
    # ------------------------------------------------------------------

    def resolve_depth(self, depth: Optional[str]) -> str:
        if not depth:
            return self.config.default_depth
        if depth not in DEPTHS:
            raise ValueError(f"Unknown depth '{depth}'. Valid: {', '.join(DEPTHS)}")
        return depth

    @staticmethod
    def _is_prose(doc) -> bool:
        """False for documents no reader can use: empty Plazi treatments, and "abstracts"
        that are CSV or tabular data (bibliography spreadsheets score high on BM25)."""
        text = ((doc.title or '') + ' ' + (doc.abstract or ''))[:500]
        if len(text.strip()) < 20:
            return False
        letters = sum(1 for c in text if c.isalpha())
        commas = text.count(',') + text.count('\t')
        if letters > 0 and commas / letters > 0.15:
            return False
        return True

    def _retrieve_collection(self, question: str, collection: str, depth: str, final_n: int,
                             candidates_n: Optional[int] = None, batch: bool = False):
        """The documents one collection gives the reader, best first.

        Returns (documents, number of candidates, info) where info holds depth_used,
        notes, queries_used and timings.
        """
        s = self.config.depth
        n = candidates_n or s.candidates_per_search
        t0 = time.perf_counter()
        info: Dict = {"depth_used": "fast", "notes": [], "queries_used": [], "timings": {}}
        timings = info["timings"]
        extra, guess_docs = [], None
        with ThreadPoolExecutor(max_workers=6) as pool:
            sub_future = None
            if depth == "deep" and self.subqueries is not None:
                # Written once per question and shared with the other collections' calls.
                sub_future = pool.submit(self.subqueries.get, question, batch)
            parsed = self.sibils.parse(question, collection)
            taxa = parsed[1]
            timings["parse"] = round(time.perf_counter() - t0, 3)
            f_today = pool.submit(self.sibils.today_list, question, collection, n, parsed)
            f_plain = pool.submit(self.sibils.plain_search, question, collection, n, taxa)
            lists, errors = [], []
            for f in (f_today, f_plain):
                try:
                    lists.append(f.result())
                except Exception as e:
                    logger.warning("a %s search failed: %s", collection, e)
                    lists.append([])
                    errors.append(e)
            if len(errors) == 2:
                raise errors[0]
            timings["search"] = round(time.perf_counter() - t0, 3)

            if depth == "deep":
                if sub_future is None:
                    info["notes"].append(note("deep_unavailable"))
                else:
                    sq = sub_future.result()
                    timings["subqueries"] = round(sq.seconds, 3)
                    if sq.failed:
                        info["notes"].append(note("depth_reduced"))
                    else:
                        info["depth_used"] = "deep"
                        t1 = time.perf_counter()
                        fq = [(q, pool.submit(self.sibils.plain_search, q, collection, n, taxa,
                                              s.extra_search_timeout)) for q in sq.queries]
                        fg = (pool.submit(self.sibils.plain_search, sq.guess_query, collection, n,
                                          taxa, s.extra_search_timeout) if sq.guess_query else None)
                        extra = [(q, [d for d in f.result() if self._is_prose(d)]) for q, f in fq]
                        guess_docs = [d for d in fg.result() if self._is_prose(d)] if fg else None
                        timings["extra_search"] = round(time.perf_counter() - t1, 3)

        base = [[d for d in lst if self._is_prose(d)] for lst in lists]
        t2 = time.perf_counter()
        sel = select(question, base, extra, guess_docs, self.reranker.score, final_n, s)
        timings["rerank"] = round(time.perf_counter() - t2, 3)
        timings["retrieval"] = round(time.perf_counter() - t0, 3)
        info["queries_used"] = sel.kept_queries
        info["dropped_queries"] = sel.dropped_queries
        info["guess_docs_in"] = sel.guess_docs_in
        info["scores"] = [round(x, 3) for x in sel.scores]
        for d, logit in zip(sel.documents, sel.scores):
            d.score = round(sigmoid(logit), 4)
        return sel.documents, sel.n_candidates, info

    def _retrieve(self, question: str, collection: Optional[str], depth: str, final_n: int,
                  candidates_n: Optional[int] = None, batch: bool = False):
        """Like _retrieve_collection; with no collection, every configured collection is
        searched and the best documents of all are kept (at most final_n // 2 from each
        collection other than Medline, as before)."""
        if collection is not None:
            return self._retrieve_collection(question, collection, depth, final_n, candidates_n, batch)
        cols = self.sibils.collection if isinstance(self.sibils.collection, list) else [self.sibils.collection]
        results, errors = {}, []
        with ThreadPoolExecutor(max_workers=len(cols)) as pool:
            futures = {c: pool.submit(self._retrieve_collection, question, c, depth, final_n, candidates_n, batch)
                       for c in cols}
            for c, f in futures.items():
                try:
                    results[c] = f.result()
                except Exception as e:
                    errors.append(e)
        if not results:
            raise errors[0]
        ranked = sorted((d for docs, _, _ in results.values() for d in docs),
                        key=lambda d: d.score, reverse=True)
        counts: Dict[str, int] = {}
        documents = []
        for d in ranked:
            limit = final_n // 2 if d.source != 'medline' else final_n
            if counts.get(d.source, 0) < limit:
                documents.append(d)
                counts[d.source] = counts.get(d.source, 0) + 1
        documents = documents[:final_n]
        infos = [info for _, _, info in results.values()]
        notes, seen = [], set()
        for i in infos:
            for nt in i["notes"]:
                if nt["code"] not in seen:
                    seen.add(nt["code"])
                    notes.append(nt)
        info = {
            "depth_used": "deep" if all(i["depth_used"] == "deep" for i in infos) else "fast",
            "notes": notes,
            "queries_used": next((i["queries_used"] for i in infos if i["queries_used"]), []),
            "timings": {k: max(i["timings"].get(k, 0.0) for i in infos)
                        for k in {k for i in infos for k in i["timings"]}},
        }
        return documents, sum(n for _, n, _ in results.values()), info

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
        clean_text = _MARKER_RUN.sub("", text).strip()
        if self._is_garbage(clean_text):
            return []
        passages = self._passages(question, documents)
        # Every document cited, in [N] or [N, M], is surfaced; none cited, the top one.
        cited_indices = sorted(set(
            int(n) for marker in re.findall(_MARKER, text) for n in re.findall(r"\d+", marker)
            if int(n) < len(documents)
        ))
        if not cited_indices:
            cited_indices = [0]
        # Return one item: the full answer text with all cited docs in the docs list.
        return [annotate({
            "answer": clean_text,
            "answer_score": None,
            "generation_confidence": getattr(raw_text, "confidence", None),
            "docs": [
                {
                    "docid": self._format_docid(documents[i]),
                    "doc_source": getattr(documents[i], 'source', ''),
                    "doc_retrieval_score": round(float(getattr(documents[i], 'score', 0.0)), 3),
                    "doc_text": (
                        (documents[i].title.strip() + ". " if documents[i].title and documents[i].title.strip() else "")
                        + (documents[i].abstract or "")
                    )[:800],
                    "snippet_start": None,
                    "snippet_end": None,
                    # The passage of the document the model was given, which is what the
                    # answer can rest on; doc_text is only the start of the abstract.
                    "passage": passages[i],
                }
                for i in cited_indices
            ],
        })]

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
            "docs": [{
                "docid": self._format_docid(doc),
                "doc_source": getattr(doc, 'source', 'sibils'),
                "doc_retrieval_score": round(float(getattr(doc, 'score', 0.0)), 3),
                "doc_text": snippet,
                "snippet_start": s - window_start,
                "snippet_end": e - window_start,
                "passage": passage,
            }],
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

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def _extractive_answers(self, question: str, documents: List, limit: Optional[int] = None) -> List[Dict]:
        candidates = self.extractor.extract(question, documents, self.config.max_abstract_length)
        if limit:
            candidates = candidates[:limit]
        return [self._build_answer_from_candidate(c, documents) for c in candidates]

    def _read(self, question: str, documents: List, mode: str, notes: List[Dict],
              extractive_limit: Optional[int] = None, debug_info: Optional[Dict] = None,
              batch: bool = False):
        """Answers from the documents, and the mode that wrote them.

        A generative answer the language model cannot give in time (no first word in
        llm_first_token_timeout seconds, server busy or down) becomes the extractive
        answer, with a note saying so: the person still gets an answer at once.
        """
        if mode == "extractive":
            return self._extractive_answers(question, documents, extractive_limit), "extractive"
        t0 = time.perf_counter()
        try:
            raw_text = self._generate(self._build_messages(question, documents), batch=batch)
        except LLMUnavailable as e:
            if batch:
                # A batch job gets the generated answer or an error, never BioBERT's span in
                # its place: a script reading the answer cannot tell the two apart.
                raise
            logger.warning("language model unavailable (%s): sending the extractive answer", e.reason)
            notes.append(note("extractive_fallback"))
            if debug_info is not None:
                debug_info['generation_error'] = str(e)
            return self._extractive_answers(question, documents, extractive_limit), "extractive"
        if debug_info is not None:
            debug_info['raw_generated_text'] = str(raw_text)
            debug_info['generation_time'] = round(time.perf_counter() - t0, 3)
        return self._answers_from_generation(question, raw_text, documents, "generative"), "generative"

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_batch(
        self,
        questions: List[str],
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> List[Dict]:
        """Run extractive QA on multiple questions, searching for all of them at once.

        Searches run concurrently across questions at the fast depth. Results are
        returned in the same order as the input list.
        """
        start_time = time.time()
        final_n = final_n or self.config.final_n

        # ── 1. Parallel retrieval ──────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=min(len(questions), 8)) as pool:
            retrieval_results = list(pool.map(
                lambda q: self._retrieve(q, collection, "fast", final_n, retrieval_n)[:2],
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
                        "docs": [{
                            "docid": self._format_docid(doc),
                            "doc_source": getattr(doc, 'source', 'sibils'),
                            "doc_retrieval_score": round(float(getattr(doc, 'score', 0.0)), 3),
                            "doc_text": cand["passage"],
                            "snippet_start": cand["span_start"],
                            "snippet_end": cand["span_end"],
                        }],
                    })
            responses.append(self._build_response(
                question, answers, documents, num_retrieved,
                final_n, collection, "extractive", time.time() - start_time,
            ))

        return responses

    def run_multi_collection(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        mode: str = "generative",
        retrieval: Optional[str] = None,
        debug: bool = False,
        depth: Optional[str] = None,
        batch: bool = False,
    ) -> Dict:
        """
        Search each collection (medline, plazi, pmc) at the depth asked, then answer per collection.

        Each collection gets its own search with final_n document slots; the deep mode's
        model-written searches are written once and used by all three. `retrieval` is
        accepted for older clients and ignored. Returns a dict with `collection_results`
        ordered best-first.
        """
        start_time = time.time()
        depth = self.resolve_depth(depth)
        final_n = final_n or self.config.final_n
        qa_collections = ["medline", "plazi", "pmc"]

        # ── 1. All searches in parallel ────────────────────────────────────
        found, errors = {}, []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {col: pool.submit(self._retrieve_collection, question, col, depth, final_n,
                                        retrieval_n, batch)
                       for col in qa_collections}
            for col, fut in futures.items():
                try:
                    found[col] = fut.result()
                except Exception as e:
                    logger.warning("search failed for %s: %s", col, e)
                    errors.append(e)
        if not found and errors:
            raise errors[0]
        total_retrieved = sum(n for _, n, _ in found.values())

        # ── 2. Collections in the order of their best document ─────────────
        # Reranker scores are comparable across collections: same question, same model.
        ranked_qa = sorted(
            [(col, docs, info) for col, (docs, _, info) in found.items() if docs],
            key=lambda item: float(getattr(item[1][0], 'score', 0.0)),
            reverse=True,
        )

        # ── 3. Read each collection's documents ───────────────────────────
        # In parallel: the shared server takes several requests at once, BioBERT runs on
        # the CPU, and an in-process model serialises its calls with its own lock.
        def read_one(item):
            col, docs, info = item
            notes = list(info["notes"])
            col_debug = {} if debug else None
            answers, mode_used = self._read(question, docs, mode, notes, extractive_limit=3,
                                            debug_info=col_debug, batch=batch)
            entry = {"collection": col, "answers": answers, "mode_used": mode_used,
                     "depth_used": info["depth_used"], "notes": notes,
                     "queries_used": info["queries_used"]}
            if debug:
                col_debug.update({k: info[k] for k in ("timings", "dropped_queries", "guess_docs_in", "scores")
                                  if k in info})
                entry['debug_info'] = col_debug
            return entry

        with ThreadPoolExecutor(max_workers=3) as pool:
            qa_results = list(pool.map(read_one, ranked_qa))
        final_mode_used = qa_results[0]["mode_used"] if qa_results else ("extractive" if mode == "extractive" else "generative")

        # ── 4. Rank collections by their answers ──────────────────────────
        # A collection whose answer is "the sources do not state it" should not be listed
        # above one that answers (D10.3 P2). The best document's score breaks the ties.
        collection_results = rank_collection_results(question, qa_results, use_scores=False)

        return {
            "question": question,
            "collection_results": collection_results,
            "mode_used": final_mode_used,
            "ndocs_retrieved": total_retrieved,
            "model": self._model_label(final_mode_used, self.config.generator_label),
            "pipeline_time": round(time.time() - start_time, 3),
            "_timings": {col: info["timings"] for col, (_, _, info) in found.items()},
            "_notes": sorted({n["code"] for r in qa_results for n in r["notes"]}),
            "_depth_used": sorted({r["depth_used"] for r in qa_results}),
        }

    def run(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        final_n: Optional[int] = None,
        collection: Optional[str] = None,
        return_documents: bool = False,
        debug: bool = False,
        mode: str = "generative",
        retrieval: Optional[str] = None,
        depth: Optional[str] = None,
        batch: bool = False,
    ) -> Dict:
        """
        Run the RAG pipeline on one collection (or on all configured ones, merged).

        Args:
            question: User question
            retrieval_n: Documents per search (default: config); the lists are merged and reranked
            final_n: Documents the reader gets (default: config, 5)
            collection: "medline", "plazi" or "pmc"; None searches every configured collection
            return_documents: Include documents in response
            debug: Include debug information
            mode: "extractive" (BioBERT span) or "generative" (language model)
            retrieval: ignored; accepted for older clients ("sparse", "dense")
            depth: "fast" or "deep" (default: config)

        Returns:
            Response dict with answers and metadata
        """
        start_time = time.time()
        depth = self.resolve_depth(depth)
        final_n = final_n or self.config.final_n
        debug_info = {} if debug else None

        documents, num_retrieved, info = self._retrieve(question, collection, depth, final_n, retrieval_n, batch)
        notes = info["notes"]
        if debug:
            debug_info['retrieval_time'] = info["timings"].get("retrieval")
            debug_info['timings'] = info["timings"]
            debug_info['initial_count'] = num_retrieved
            debug_info['final_count'] = len(documents)
            for k in ("dropped_queries", "guess_docs_in", "scores"):
                if k in info:
                    debug_info[k] = info[k]

        answers: List[Dict] = []
        mode_used = "extractive" if mode == "extractive" else "generative"
        if documents:
            t0 = time.perf_counter()
            answers, mode_used = self._read(question, documents, mode, notes, debug_info=debug_info,
                                            batch=batch)
            info["timings"]["read"] = round(time.perf_counter() - t0, 3)

        response = {
            'sibils_version': PIPELINE_VERSION,
            'success': True,
            'error': '',
            'question': question,
            'collection': collection or self._default_collection_str,
            'model': self._model_label(mode_used, self.config.generator_label),
            'ndocs_requested': final_n,
            'ndocs_returned_by_SIBiLS': num_retrieved,
            'answers': answers,
            'mode_used': mode_used,
            'depth_used': info["depth_used"],
            'notes': notes,
            'queries_used': info["queries_used"],
            'pipeline_time': round(time.time() - start_time, 3),
            'transformed_query': None,
            '_timings': info["timings"],
        }

        if debug:
            response['debug_info'] = debug_info

        if return_documents:
            response['documents'] = [
                {
                    'docid': self._format_docid(doc),
                    'source': getattr(doc, 'source', 'sibils'),
                    'title': doc.title,
                    'abstract': doc.abstract,
                    'pmid': getattr(doc, 'pmid', None),
                    'pmcid': getattr(doc, 'pmcid', None),
                    'doi': getattr(doc, 'doi', None),
                } for doc in documents
            ]

        return response

    def run_with_contexts(
        self,
        question: str,
        contexts: List[str],
        mode: str = "extractive",
        batch: bool = False,
    ) -> Dict:
        """
        Run QA on provided context strings, bypassing retrieval entirely.

        Used for held-out evaluation where the gold context is known in advance
        (e.g. BioASQ QA-only benchmark). Contexts are wrapped as Document objects
        and fed directly into the extractive or generative QA step.
        """
        from .retrieval.sibils_retriever import Document

        docs = [
            Document(doc_id=f"ctx_{i}", title="", abstract=ctx, score=1.0, source="gold")
            for i, ctx in enumerate(contexts)
        ]

        start_time = time.time()
        notes: List[Dict] = []
        answers, mode_used = self._read(question, docs, mode, notes, batch=batch)

        return {
            "question": question,
            "answers": answers,
            "mode_used": mode_used,
            "notes": notes,
            "pipeline_time": round(time.time() - start_time, 3),
        }

    def run_with_documents(
        self,
        question: str,
        documents: List,
        mode: str = "generative",
        return_documents: bool = False,
        debug: bool = False,
        batch: bool = False,
    ) -> Dict:
        """
        Run QA on pre-fetched Document objects, bypassing retrieval entirely.

        Unlike run_with_contexts(), this preserves full Document metadata
        (title, full_text, pmid, source, etc.) so the response includes
        correct doc_source, docid, and full-text IDF passage selection.
        """
        start_time = time.time()
        notes: List[Dict] = []
        answers, mode_used = self._read(question, documents, mode, notes, batch=batch)

        result: Dict = {
            "sibils_version": PIPELINE_VERSION,
            "success": True,
            "error": "",
            "question": question,
            "collection": "+".join(sorted({d.source for d in documents})),
            "model": self._model_label(mode_used, self.config.generator_label),
            "ndocs_requested": len(documents),
            "ndocs_returned_by_SIBiLS": len(documents),
            "answers": answers,
            "mode_used": mode_used,
            "depth_used": None,
            "notes": notes,
            "queries_used": [],
            "pipeline_time": round(time.time() - start_time, 3),
            "transformed_query": None,
            "debug_info": {"doc_ids": [d.doc_id for d in documents]} if debug else None,
        }
        if return_documents:
            result["documents"] = [
                {"docid": d.doc_id, "title": d.title, "source": d.source, "pmid": d.pmid, "pmcid": d.pmcid}
                for d in documents
            ]
        return result

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

    @staticmethod
    def _best_passage(text: str, question: str, window: int) -> str:
        """Anchored passage selection, shared with the extractive reader (passage.best_passage)."""
        return best_passage(text, question, window)

    def _passages(self, question: str, documents: List) -> List[str]:
        """The passage of each document the language model is given, cleaned.

        A document of up to llm_whole_upto characters is given whole: cutting abstracts
        removed the answer text for 5 of the 258 test questions. A longer one (a PMC full
        text, a long treatment) gives its best llm_abstract_length window.
        """
        out = []
        for doc in documents:
            text = doc.full_text or doc.abstract or ''
            if len(text) > self.config.llm_whole_upto:
                text = self._best_passage(text, question, self.config.llm_abstract_length)
            out.append(self._clean_for_llm(text))
        return out

    def _generate(self, messages: List[dict], batch: bool = False) -> str:
        """The answer text, from the backend the configuration names.
        Raises LLMUnavailable when the shared server does not answer in time."""
        if self.remote is not None:
            return self._generate_remote(messages, batch)
        if self.config.use_vllm:
            return self._generate_vllm(messages)
        return self._generate_cpu(messages)

    def _generate_remote(self, messages: List[dict], batch: bool = False) -> str:
        """One answer from the shared server, with the settings the reader was tested with:
        no repetition penalty (at 1.15 the 27B answered "not stated" on passages that hold
        the value), at most max_tokens tokens, no thinking. A blank answer is asked for
        once more, with another seed."""
        kwargs = dict(
            max_tokens=self.config.max_tokens, temperature=self.config.temperature,
            top_p=0.9, top_k=0, repetition_penalty=1.0, presence_penalty=0.0,
            stop=ANSWER_STOP, prefix=ANSWER_PREFIX, logprobs=True,
        )
        if batch:   # behind people's requests on the shared server, and allowed to wait
            kwargs.update(priority=self.config.llm_batch_priority,
                          first_token_timeout=self.config.llm_batch_first_token_timeout,
                          total_timeout=self.config.llm_batch_total_timeout)
        comp = self.remote.chat(messages, seed=0, **kwargs)
        if not comp.text.strip():
            comp = self.remote.chat(messages, seed=1, **kwargs)
        return GeneratedText(ANSWER_PREFIX + comp.text.strip(), mean_token_probability(comp.logprobs))

    def _build_messages(self, question: str, documents: List) -> List[dict]:
        """Build chat messages for the LLM from question and retrieved documents.

        Returns a list of message dicts (system + user) for use with llm.chat().
        Uses llm_abstract_length (shorter than BioBERT's max_abstract_length) to
        stay well within vLLM's max_model_len=4096.

        For PMC documents with full_text, uses a keyword-window scan to find the
        most relevant passage rather than blindly truncating from the start — this
        surfaces species descriptions, measurement tables, etc. that are buried deep
        in the full text but not present in the abstract.

        `/no_think` at the end of the user message is Qwen3's soft switch for
        disabling chain-of-thought reasoning at the tokenizer/template level.
        This works regardless of vLLM version, unlike chat_template_kwargs.
        """
        passages = self._passages(question, documents)
        context_parts = []
        for i, doc in enumerate(documents):
            title = self._clean_for_llm(doc.title or '')
            context_parts.append(f"[{i}] {title}\n{passages[i]}")

        context = "\n\n".join(context_parts)
        # D10.3 P4: in 14 of 25 body-size cases answered "not stated", the value was in
        # the passage the model was given. D10.3 P6: habitat answers listed collection
        # localities and countries as if they were habitats.
        system = (
            "You are a biomedical expert assistant. Answer the question based ONLY "
            "on the provided scientific sources. Be concise and factual (2-4 sentences). "
            "Cite sources using [0], [1], etc. after each claim. "
            "When a source states the requested value, such as a measurement with its "
            "units, give it as the source states it and name the organism it refers to. "
            "Say the sources lack the information only if none of them contains it, "
            "and then say so briefly. "
            "Always respond in English only."
        )
        if _HABITAT_Q.search(question or ""):
            system += (
                " For a habitat question, describe the environment and substrate "
                "(for example soil, leaf litter, moss, dead wood, caves, forest, grassland, "
                "seashore), not the countries, regions or localities where specimens "
                "were collected."
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
            logprobs=1,   # the sampled token's log-probability, for generation_confidence
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
        completion = outputs[0].outputs[0]
        continuation = completion.text.strip()
        return GeneratedText(ENGLISH_PREFIX + continuation, self._confidence(completion))

    @staticmethod
    def _confidence(completion) -> Optional[float]:
        """Mean probability of the generated tokens (geometric), or None."""
        try:
            lps = []
            for tok, entry in zip(completion.token_ids, completion.logprobs or []):
                lp = entry.get(tok) if isinstance(entry, dict) else None
                if lp is not None:
                    lps.append(getattr(lp, "logprob", lp))
            return mean_token_probability(lps)
        except Exception:
            return None

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

