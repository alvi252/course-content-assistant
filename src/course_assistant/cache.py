"""Embedding cache on top of any provider: each distinct text is embedded at most once."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingProvider
from .store import Store
from .usage import UsageLog


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedEmbedding:
    vectors: np.ndarray
    computed: int  # texts sent to the provider
    from_cache: int  # texts served from disk
    tokens: int  # tokens billed by the provider for this call


class CachedEmbedder:
    def __init__(
        self, provider: EmbeddingProvider, store: Store, usage_log: UsageLog | None = None
    ) -> None:
        self.provider = provider
        self.store = store
        self.usage_log = usage_log or UsageLog(None)

    @property
    def model(self) -> str:
        return self.provider.model

    def embed(self, texts: Sequence[str]) -> CachedEmbedding:
        if not texts:
            return CachedEmbedding(np.zeros((0, 0), dtype=np.float32), 0, 0, 0)
        hashes = [text_hash(t) for t in texts]
        cached = self.store.cached_embeddings(self.model, sorted(set(hashes)))

        missing: dict[str, str] = {}
        for h, t in zip(hashes, texts, strict=True):
            if h not in cached:
                missing.setdefault(h, t)

        tokens = 0
        if missing:
            result = self.provider.embed(list(missing.values()))
            tokens = result.tokens
            fresh = dict(zip(missing.keys(), result.vectors, strict=True))
            self.store.cache_embeddings(self.model, fresh)
            cached.update(fresh)
            self.usage_log.record("embedding", self.model, tokens)

        vectors = np.vstack([cached[h] for h in hashes]).astype(np.float32)
        return CachedEmbedding(
            vectors=vectors,
            computed=len(missing),
            from_cache=len(texts) - sum(1 for h in hashes if h in missing),
            tokens=tokens,
        )
