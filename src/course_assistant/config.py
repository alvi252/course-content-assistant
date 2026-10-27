"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw and raw.strip() else default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


@dataclass(frozen=True)
class Settings:
    """All tunables in one place. Every field maps to an environment variable."""

    author_name: str = "Mira Calder"
    docs_dir: Path = Path("data/library")
    voice_dir: Path = Path("data/voice")
    index_path: Path = Path("data/index.sqlite")

    # Providers
    llm_provider: str = "openai"  # "openai" | "anthropic"
    llm_model: str = "gpt-4.1-mini"
    embedding_provider: str = "openai"  # "openai" | "hashing" (offline, for tests/demos)
    embedding_model: str = "text-embedding-3-small"
    request_timeout_s: float = 60.0

    # Chunking
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    # Retrieval
    top_k: int = 5
    min_similarity: float = 0.30
    hybrid: bool = True

    # Ideas
    ideas_default_count: int = 15
    ideas_max_count: int = 30
    ideas_per_lesson: int = 2  # max passages from one lesson in a single request
    min_passage_chars: int = 200  # shorter chunks are not worth an idea
    skip_used_days: int = 30  # passages used within this many days are skipped (0 = never skip)
    ideas_temperature: float = 0.7
    ideas_tokens_per_idea: int = 250  # output budget: 400 + this * passages in the request
    ideas_seed: int | None = None  # set it and selection tie-breaks become deterministic

    # Scripts
    script_temperature: float = 0.6
    script_max_tokens: int = 900

    # Operations
    admin_token: str = field(default="", repr=False)
    usage_log_path: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        usage_log = os.environ.get("USAGE_LOG_PATH", "").strip()
        provider = _env_str("LLM_PROVIDER", cls.llm_provider).lower()
        default_model = cls.llm_model if provider == "openai" else "claude-haiku-5-5"
        return cls(
            author_name=_env_str("AUTHOR_NAME", cls.author_name),
            docs_dir=Path(_env_str("LIBRARY_DIR", str(cls.docs_dir))),
            voice_dir=Path(_env_str("VOICE_DIR", str(cls.voice_dir))),
            index_path=Path(_env_str("INDEX_PATH", str(cls.index_path))),
            llm_provider=provider,
            llm_model=_env_str("LLM_MODEL", default_model),
            embedding_provider=_env_str("EMBEDDING_PROVIDER", cls.embedding_provider).lower(),
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", cls.request_timeout_s),
            chunk_max_chars=_env_int("CHUNK_MAX_CHARS", cls.chunk_max_chars),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", cls.chunk_overlap_chars),
            top_k=_env_int("TOP_K", cls.top_k),
            min_similarity=_env_float("MIN_SIMILARITY", cls.min_similarity),
            hybrid=_env_bool("HYBRID_SEARCH", cls.hybrid),
            ideas_default_count=_env_int("IDEAS_DEFAULT_COUNT", cls.ideas_default_count),
            ideas_max_count=_env_int("IDEAS_MAX_COUNT", cls.ideas_max_count),
            ideas_per_lesson=_env_int("IDEAS_PER_LESSON", cls.ideas_per_lesson),
            min_passage_chars=_env_int("MIN_PASSAGE_CHARS", cls.min_passage_chars),
            skip_used_days=_env_int("SKIP_USED_DAYS", cls.skip_used_days),
            ideas_temperature=_env_float("IDEAS_TEMPERATURE", cls.ideas_temperature),
            ideas_tokens_per_idea=_env_int("IDEAS_TOKENS_PER_IDEA", cls.ideas_tokens_per_idea),
            ideas_seed=int(seed) if (seed := os.environ.get("IDEAS_SEED", "").strip()) else None,
            script_temperature=_env_float("SCRIPT_TEMPERATURE", cls.script_temperature),
            script_max_tokens=_env_int("SCRIPT_MAX_TOKENS", cls.script_max_tokens),
            admin_token=os.environ.get("ADMIN_TOKEN", "").strip(),
            usage_log_path=Path(usage_log) if usage_log else None,
        )


def load_dotenv_if_present(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env without overriding variables already set."""
    if not path.is_file():
        return
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
