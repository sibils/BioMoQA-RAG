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
    max_tokens: int = 384
    temperature: float = 0.1


class RetrievalConfig(BaseModel):
    retrieval_n: int = 15
    use_smart_retrieval: bool = True
    hybrid_alpha: float = 0.5
    # Dense retriever encoder, used by the hybrid (generative) path.
    dense_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"


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
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 15


class RelevanceFilterConfig(BaseModel):
    enabled: bool = True
    min_overlap: float = 0.15
    final_n: int = 5
    min_extractive_score: float = 0.20


class ContextConfig(BaseModel):
    max_abstract_length: int = 800
    llm_abstract_length: int = 600
    truncate_abstracts: bool = True


class DataConfig(BaseModel):
    faiss_index: str = "data/faiss_index.bin"
    documents: str = "data/documents.pkl"


class ExtractionConfig(BaseModel):
    device: int = 0  # 0 = GPU, -1 = CPU
    # BioBERT extractive QA model. Lazy-loaded on first extractive question.
    model: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    sibils: SibilsConfig = SibilsConfig()
    reranking: RerankingConfig = RerankingConfig()
    relevance_filter: RelevanceFilterConfig = RelevanceFilterConfig()
    context: ContextConfig = ContextConfig()
    data: DataConfig = DataConfig()
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

    The reranker is included whether or not reranking is enabled, because the
    pipeline loads it at construction time regardless, and an unused 1 GB download
    is cheaper than a failed deploy.
    """
    cfg = config or get_config()
    models = [
        cfg.model.model_name,
        cfg.reranking.model,
        cfg.extraction.model,
        cfg.retrieval.dense_model,
    ]
    return list(dict.fromkeys(m for m in models if m))


def print_required_models() -> None:
    """Entry point: print one HuggingFace repo id per line, for the deploy playbook."""
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    for model in required_models(get_config(path) if path else None):
        print(model)
