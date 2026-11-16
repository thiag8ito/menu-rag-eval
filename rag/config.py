import os
from dataclasses import dataclass, field


def _env(name: str, default: str):
    # default_factory: lê o ambiente na hora de instanciar, não no import do módulo
    return field(default_factory=lambda: os.getenv(name, default))


def _env_int(name: str, default: int):
    return field(default_factory=lambda: int(os.getenv(name, str(default))))


def _env_bool(name: str, default: bool):
    return field(default_factory=lambda: os.getenv(name, "1" if default else "0") == "1")


@dataclass(frozen=True)
class Settings:
    # defaults = config escolhida no dev pelo experimento offline (ver results/report.md)
    data_dir: str = _env("DATA_DIR", "data")
    index_dir: str = _env("INDEX_DIR", "artifacts/index")
    embedder: str = _env("EMBEDDER", "tfidf-char")
    reranker: str = _env("RERANKER", "none")
    chunk_strategy: str = _env("CHUNK_STRATEGY", "structured")
    chunk_size: int = _env_int("CHUNK_SIZE", 1024)
    contextual_header: bool = _env_bool("CONTEXTUAL_HEADER", True)
    hybrid: bool = _env_bool("HYBRID", True)
    entity_link: bool = _env_bool("ENTITY_LINK", True)
    candidates_k: int = _env_int("CANDIDATES_K", 20)
    generator_model: str = _env("GENERATOR_MODEL", "claude-haiku-5-5")
    judge_model: str = _env("JUDGE_MODEL", "claude-sonnet-5-5")


def get_settings() -> Settings:
    return Settings()
