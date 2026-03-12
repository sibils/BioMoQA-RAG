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


class GenerationConfig(BaseModel):
    max_tokens: int = 384
    temperature: float = 0.1


class RetrievalConfig(BaseModel):
    retrieval_n: int = 15
    use_smart_retrieval: bool = True
    hybrid_alpha: float = 0.5


class SibilsConfig(BaseModel):
    search_api_url: str = "https://biodiversitypmc.sibils.org/api/search"
    query_parser_api_url: str = "https://biodiversitypmc.dev.sibils.org/api/query/parse"
    collections: List[str] = ["medline", "plazi"]
    user_agent: str = "BioMoQA-RAG/1.0 (https://github.com/sibils/BioMoQA-RAG)"
    cache_dir: str = "data/sibils_cache"
    cache_ttl: int = 604800  # 7 days


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 15


class RelevanceFilterConfig(BaseModel):
    enabled: bool = True
    min_overlap: float = 0.15
    final_n: int = 5


class ContextConfig(BaseModel):
    max_abstract_length: int = 800
    truncate_abstracts: bool = True


class DataConfig(BaseModel):
    faiss_index: str = "data/faiss_index.bin"
    documents: str = "data/documents.pkl"


class ExtractionConfig(BaseModel):
    device: int = 0  # 0 = GPU, -1 = CPU


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
