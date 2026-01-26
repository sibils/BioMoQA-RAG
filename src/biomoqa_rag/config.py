"""
Configuration management for BioMoQA RAG.

Reads configuration from config.toml file.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Use tomllib (Python 3.11+) or tomli (Python 3.9-3.10)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class Config:
    """Configuration loaded from config.toml."""

    _instance: Optional["Config"] = None
    _config: dict = {}

    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern - only load config once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: Optional[str] = None):
        """Load configuration from TOML file."""
        if config_path:
            path = Path(config_path)
        else:
            # Default: ./config.toml
            path = Path("config.toml")

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "rb") as f:
            self._config = tomllib.load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation or multiple keys.

        Examples:
            config.get("server", "port")  # -> 9000
            config.get("model", "mode")   # -> "gpu"
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def server(self) -> dict:
        """Server configuration."""
        return self._config.get("server", {})

    @property
    def model(self) -> dict:
        """Model configuration."""
        return self._config.get("model", {})

    @property
    def generation(self) -> dict:
        """Generation configuration."""
        return self._config.get("generation", {})

    @property
    def retrieval(self) -> dict:
        """Retrieval configuration."""
        return self._config.get("retrieval", {})

    @property
    def reranking(self) -> dict:
        """Reranking configuration."""
        return self._config.get("reranking", {})

    @property
    def relevance_filter(self) -> dict:
        """Relevance filter configuration."""
        return self._config.get("relevance_filter", {})

    @property
    def context(self) -> dict:
        """Context configuration."""
        return self._config.get("context", {})

    @property
    def data(self) -> dict:
        """Data paths configuration."""
        return self._config.get("data", {})


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the configuration singleton."""
    return Config(config_path)
