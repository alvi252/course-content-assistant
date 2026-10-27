"""Wire the components together from settings (the only place that knows about all of them)."""

from __future__ import annotations

from dataclasses import dataclass

from .cache import CachedEmbedder
from .config import Settings
from .embeddings import EmbeddingProvider, build_embedding_provider
from .ideas import IdeaService
from .ingest import Ingestor
from .llm import LazyLLM, LLMProvider
from .retrieval import Retriever
from .store import Store
from .usage import UsageLog


@dataclass
class Services:
    settings: Settings
    store: Store
    embedder: CachedEmbedder
    ingestor: Ingestor
    retriever: Retriever
    ideas: IdeaService


def build_ingestor(
    settings: Settings, *, embedding_provider: EmbeddingProvider | None = None
) -> Ingestor:
    """Ingestion needs a store and an embedder, but no chat model."""
    usage_log = UsageLog(settings.usage_log_path)
    store = Store(settings.index_path)
    provider = embedding_provider or build_embedding_provider(
        settings.embedding_provider, settings.embedding_model, settings.request_timeout_s
    )
    return Ingestor(
        store,
        CachedEmbedder(provider, store, usage_log),
        settings.docs_dir,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )


def build_services(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
) -> Services:
    """Build everything; providers can be injected (tests, custom backends).

    The chat model is created on its first use, so the app starts without an LLM key.
    """
    ingestor = build_ingestor(settings, embedding_provider=embedding_provider)
    store, embedder = ingestor.store, ingestor.embedder
    retriever = Retriever(
        store,
        embedder,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        hybrid=settings.hybrid,
    )
    chat = llm or LazyLLM(settings.llm_provider, settings.llm_model, settings.request_timeout_s)
    ideas = IdeaService(store, retriever, chat, settings, usage_log=embedder.usage_log)
    return Services(settings, store, embedder, ingestor, retriever, ideas)
