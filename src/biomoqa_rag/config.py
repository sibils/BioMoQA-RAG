"""
Configuration management for BioMoQA RAG.

Uses Pydantic BaseModel to validate and type-check config.toml values.
"""

import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9000
    workers: int = 1
    log_level: str = "info"


class ModelConfig(BaseModel):
    mode: str = "gpu"
    size: str = "8b"
    model_name: str = "Qwen/Qwen3-8B"
    gpu_memory_utilization: float = 0.83
    quantization: Optional[str] = "fp8"
    # Context window vLLM allocates KV cache for. Raise only with the VRAM to back
    # it: KV cache for Qwen3-8B costs ~144 KB per token.
    max_model_len: int = 4096


class GenerationConfig(BaseModel):
    # "remote": the shared language-model server ([llm]); "vllm": the [model] loaded in
    # this process on the GPU; "cpu": a small model on the CPU (model.size).
    backend: str = "remote"
    max_tokens: int = 300
    temperature: float = 0.2


class LLMConfig(BaseModel):
    """The shared language-model server (OpenAI-compatible; LAN only, no key)."""
    base_url: str = "http://llm.lan.text-analytics.ch/qwen/v1"
    model: str = "Qwen3.8-27B"
    # No first token in this time: the extractive answer is sent instead, with a note.
    first_token_timeout: float = 4.0
    total_timeout: float = 20.0
    priority: int = -10  # interactive; batch jobs on the same server send a positive value
    # Batch jobs (a POST that does not come from a web browser, such as the trait pipeline):
    # served after people, allowed to wait, and an error rather than BioBERT's answer.
    batch_priority: int = 10
    batch_first_token_timeout: float = 60.0
    batch_total_timeout: float = 100.0


class DepthConfig(BaseModel):
    """Search depth (see depth.py). Requests choose with `depth`; this is the default."""
    default: str = "fast"
    candidates_per_search: int = 20
    subquery_timeout: float = 3.0
    extra_search_timeout: float = 1.0
    min_agreement: float = 0.8
    max_new_words: int = 5
    guess_query: bool = True
    guess_margin: float = 1.0
    batch_subquery_timeout: float = 30.0


class RetrievalConfig(BaseModel):
    # Not used since the depth search ([depth] candidates_per_search); kept so that
    # older config files still load.
    retrieval_n: int = 20


class SibilsConfig(BaseModel):
    search_api_url: str = "https://biodiversitypmc.sibils.org/api/search"
    query_parser_api_url: str = "https://biodiversitypmc.dev.sibils.org/api/query/parse"
    collections: List[str] = ["medline", "plazi"]
    user_agent: str = "BioMoQA-RAG/1.0 (https://github.com/sibils/BioMoQA-RAG)"
    cache_dir: str = "data/sibils_cache"
    cache_ttl: int = 604800  # 7 days
    empty_cache_ttl: int = 0  # never serve empty results from cache (0-result responses are not cached)
    enforce_taxon_phrase: bool = True  # require recognised binomials as a phrase (avoids epithet-only matches)


class RerankingConfig(BaseModel):
    # Every candidate is scored against the question as asked; the best final_n are read.
    model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    device: str = "cpu"


class RelevanceFilterConfig(BaseModel):
    final_n: int = 5                     # documents the reader gets
    min_extractive_score: float = 0.20   # POST /batch drops extractive answers below this


class ContextConfig(BaseModel):
    max_abstract_length: int = 800
    llm_abstract_length: int = 600   # window the language model gets from a long document
    llm_whole_upto: int = 2500       # documents up to this length are given whole
    truncate_abstracts: bool = True


class ExtractionConfig(BaseModel):
    device: int = 0  # 0 = GPU, -1 = CPU
    # BioBERT extractive QA model. Lazy-loaded on first extractive question.
    model: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    llm: LLMConfig = LLMConfig()
    depth: DepthConfig = DepthConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    sibils: SibilsConfig = SibilsConfig()
    reranking: RerankingConfig = RerankingConfig()
    relevance_filter: RelevanceFilterConfig = RelevanceFilterConfig()
    context: ContextConfig = ContextConfig()
    extraction: ExtractionConfig = ExtractionConfig()


_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Load and return the configuration singleton."""
    global _config
    if _config is None:
        path = Path(config_path) if config_path else Path("config.toml")
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        _config = Config(**raw)
    return _config


def required_models(config: Optional[Config] = None) -> List[str]:
    """Every HuggingFace repo id the running service resolves through the Hub.

    The service runs with HF_HUB_OFFLINE=1 and downloads nothing, so the deployment
    must pre-populate the cache with exactly this list. Deriving it from the config
    rather than restating it in the playbook is what stops the two drifting apart:
    a model changed in config.toml is downloaded on the next deploy without anyone
    having to remember a second place to edit.

    The generative model is included only when this process loads it (backend
    "vllm" or "cpu"); with the shared server it is not needed here.
    """
    cfg = config or get_config()
    models = [
        cfg.model.model_name if cfg.generation.backend != "remote" else None,
        cfg.reranking.model,
        cfg.extraction.model,
    ]
    return list(dict.fromkeys(m for m in models if m))


def print_required_models() -> None:
    """Entry point: print one HuggingFace repo id per line, for the deploy playbook."""
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    for model in required_models(get_config(path) if path else None):
        print(model)
