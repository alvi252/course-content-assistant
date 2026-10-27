"""Vector search, optional BM25 keyword search, and reciprocal rank fusion."""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .cache import CachedEmbedder
from .errors import IndexModelMismatchError
from .models import Chunk, RetrievedChunk
from .store import Store

RRF_K = 60  # standard constant from Cormack et al. (2009); dampens the weight of top ranks

_TOKEN = re.compile(r"\w+", re.UNICODE)
# fmt: off
_STOPWORDS = frozenset((
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has",
    "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so",
    "that", "the", "their", "them", "then", "there", "this", "to", "us", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
    # French and German function words (the sample questions are not all in English)
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "que", "qui", "pour", "par",
    "sur", "dans", "au", "aux", "ce", "je", "mon", "ma", "mes", "vous", "votre", "nous",
    "der", "die", "das", "und", "ist", "ich", "sie", "ein", "eine", "zu",
))
# fmt: on


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords, with light plural folding.

    >>> tokenize("What are the Delivery times for parcels?")
    ['delivery', 'time', 'parcel']
    """
    tokens = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        if len(raw) > 4 and raw.endswith("ies"):
            raw = raw[:-3] + "y"
        elif len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        tokens.append(raw)
    return tokens


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Fuse ranked lists of item ids: score(id) = sum over lists of 1 / (k + rank).

    >>> scores = reciprocal_rank_fusion([[7, 3], [3, 9]], k=60)
    >>> max(scores, key=scores.get)
    3
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass(frozen=True)
class SearchResult:
    chunks: list[RetrievedChunk]
    best_similarity: float  # highest cosine similarity in the whole index, before the threshold
    embedding_tokens: int


@dataclass(frozen=True)
class _Index:
    generation: int
    chunks: list[Chunk]
    matrix: np.ndarray
    model: str | None
    bm25: BM25Okapi | None


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: CachedEmbedder,
        *,
        top_k: int = 5,
        min_similarity: float = 0.30,
        hybrid: bool = True,
        candidate_pool: int = 20,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.hybrid = hybrid
        self.candidate_pool = candidate_pool
        self._index: _Index | None = None
        self._chunks: tuple[int, list[Chunk]] | None = None
        self._lock = threading.Lock()

    def _current_index(self) -> _Index:
        """Load the index once and reload it only after an ingest changed something."""
        generation = self.store.generation()
        with self._lock:
            if self._index is None or self._index.generation != generation:
                chunks, matrix, model = self.store.load_index()
                bm25 = None
                if chunks and self.hybrid:
                    corpus = [tokenize(f"{c.doc_title} {c.heading} {c.text}") for c in chunks]
                    bm25 = BM25Okapi(corpus)
                self._index = _Index(generation, chunks, matrix, model, bm25)
            return self._index

    def all_chunks(self) -> list[Chunk]:
        """Every indexed chunk in document order: no embedding call, no vectors, no model check.

        Cached until an ingest bumps the store's generation.
        """
        generation = self.store.generation()
        with self._lock:
            if self._chunks is None or self._chunks[0] != generation:
                self._chunks = (generation, self.store.load_chunks())
            return self._chunks[1]

    def search(self, query: str, top_k: int | None = None) -> SearchResult:
        k = top_k or self.top_k
        index = self._current_index()
        if not index.chunks:
            return SearchResult([], 0.0, 0)
        if index.model != self.embedder.model:
            raise IndexModelMismatchError(
                f"The index was built with {index.model!r} but the configured embedding model "
                f"is {self.embedder.model!r}. Run ingest --rebuild."
            )

        embedded = self.embedder.embed([query])
        similarities = index.matrix @ embedded.vectors[0]
        best = float(similarities.max())

        pool = min(max(self.candidate_pool, k), len(index.chunks))
        vector_ranking = [int(i) for i in np.argsort(-similarities)[:pool]]
        vector_rank = {item: rank for rank, item in enumerate(vector_ranking, start=1)}

        keyword_rank: dict[int, int] = {}
        if index.bm25 is not None:
            query_tokens = tokenize(query)
            if query_tokens:
                bm25_scores = index.bm25.get_scores(query_tokens)
                keyword_ranking = [
                    int(i) for i in np.argsort(-bm25_scores)[:pool] if bm25_scores[i] > 0
                ]
                keyword_rank = {item: rank for rank, item in enumerate(keyword_ranking, start=1)}
            fused = reciprocal_rank_fusion([vector_ranking, list(keyword_rank)])
        else:
            fused = {item: float(similarities[item]) for item in vector_ranking}

        # The similarity threshold is the relevance gate: keyword matches can re-rank
        # semantically relevant chunks, but cannot pull in chunks the embedding deems unrelated.
        candidates = [i for i in fused if similarities[i] >= self.min_similarity]
        candidates.sort(key=lambda i: (fused[i], similarities[i]), reverse=True)

        results = [
            RetrievedChunk(
                chunk=index.chunks[i],
                similarity=float(similarities[i]),
                score=float(fused[i]),
                vector_rank=vector_rank.get(i),
                keyword_rank=keyword_rank.get(i),
            )
            for i in candidates[:k]
        ]
        return SearchResult(results, best, embedded.tokens)
