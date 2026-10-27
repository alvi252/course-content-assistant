"""Embedding providers. Vectors are always returned L2-normalised as float32."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .errors import ProviderError

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # shape (n, dim), float32, unit length
    tokens: int


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class OpenAIEmbeddings:
    """OpenAI embeddings (default ``text-embedding-3-small``), batched."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        timeout_s: float = 30.0,
        batch_size: int = 96,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._client = client  # created on first use, so the app starts without a key

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            from .llm import make_client

            def build() -> Any:
                from openai import OpenAI

                return OpenAI(timeout=self._timeout_s, max_retries=2)  # reads OPENAI_API_KEY

            self._client = make_client("OpenAI", build)
        return self._client

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(np.zeros((0, 0), dtype=np.float32), 0)
        client = self._get_client()
        rows: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = client.embeddings.create(model=self._model, input=batch)
            except Exception as exc:
                raise ProviderError("OpenAI embeddings", exc) from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            rows.extend(item.embedding for item in ordered)
            usage = getattr(response, "usage", None)
            tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        return EmbeddingResult(normalize_rows(np.array(rows, dtype=np.float32)), tokens)


class HashingEmbeddings:
    """Deterministic, offline bag-of-words embeddings (feature hashing).

    No semantic understanding: words must overlap to match. It exists so the test-suite and
    offline demos run without network access or API keys.
    """

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def model(self) -> str:
        return f"hashing-{self._dim}"

    def _index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self._dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        tokens = 0
        for row, text in enumerate(texts):
            words = [w for w in _WORD.findall(text.lower()) if len(w) > 2]
            tokens += len(words)
            for word in words:
                index, sign = self._index(word)
                matrix[row, index] += sign
        return EmbeddingResult(normalize_rows(matrix), tokens)


def build_embedding_provider(name: str, model: str, timeout_s: float) -> EmbeddingProvider:
    if name == "openai":
        return OpenAIEmbeddings(model, timeout_s=timeout_s)
    if name == "hashing":
        return HashingEmbeddings()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {name!r} (expected 'openai' or 'hashing')")
