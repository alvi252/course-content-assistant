"""Idempotent ingestion: only new or changed files are chunked and embedded."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .cache import CachedEmbedder
from .chunking import chunk_document
from .history import rebind_ideas
from .loaders import UnsupportedFileError, iter_source_files, load_document
from .models import IngestReport
from .store import Store

logger = logging.getLogger(__name__)

# Bump when chunking logic changes in a way that should invalidate existing chunks.
CHUNKER_VERSION = "1"


def document_fingerprint(
    content_hash: str, embedding_model: str, max_chars: int, overlap_chars: int
) -> str:
    """Everything that, if changed, makes the stored chunks stale."""
    key = f"{content_hash}|{embedding_model}|{max_chars}|{overlap_chars}|v{CHUNKER_VERSION}"
    return hashlib.sha256(key.encode()).hexdigest()


class Ingestor:
    def __init__(
        self,
        store: Store,
        embedder: CachedEmbedder,
        docs_dir: Path,
        *,
        max_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.docs_dir = docs_dir
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def run(self, *, rebuild: bool = False) -> IngestReport:
        if not self.docs_dir.is_dir():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        report = IngestReport()
        existing = self.store.fingerprints()

        seen: set[str] = set()
        for path in iter_source_files(self.docs_dir):
            rel = path.relative_to(self.docs_dir).as_posix()
            try:
                doc = load_document(path, self.docs_dir)
            except (UnsupportedFileError, OSError, ValueError) as exc:
                logger.warning("Skipping %s: %s", rel, exc)
                report.skipped.append(rel)
                if rel in existing:
                    seen.add(rel)  # unreadable right now: keep the last good version
                continue

            fingerprint = document_fingerprint(
                doc.content_hash, self.embedder.model, self.max_chars, self.overlap_chars
            )
            # A rebuild re-processes every file; chunks are swapped per document, so the
            # index stays fully searchable while it runs (and the cache avoids re-billing).
            if not rebuild and existing.get(doc.path) == fingerprint:
                seen.add(doc.path)
                report.unchanged.append(doc.path)
                continue

            chunks = chunk_document(doc, self.max_chars, self.overlap_chars)
            if not chunks:
                logger.warning("Skipping %s: no extractable text", rel)
                report.skipped.append(rel)
                continue

            embedded = self.embedder.embed([c.embed_text for c in chunks])
            self.store.replace_document(
                path=doc.path,
                title=doc.title,
                fingerprint=fingerprint,
                model=self.embedder.model,
                chunks=chunks,
                vectors=embedded.vectors,
            )
            seen.add(doc.path)
            (report.updated if doc.path in existing else report.added).append(doc.path)
            report.chunks_written += len(chunks)
            report.embeddings_computed += embedded.computed
            report.embeddings_from_cache += embedded.from_cache
            report.embedding_tokens += embedded.tokens

        stale = sorted(set(existing) - seen)
        if stale:
            self.store.delete_documents(stale)
            report.removed = stale

        if report.changed:
            rebind_ideas(self.store)
            self.store.bump_generation()
        logger.info(
            "Ingest done: %d added, %d updated, %d unchanged, %d removed, %d skipped;"
            " %d embeddings computed, %d from cache",
            len(report.added),
            len(report.updated),
            len(report.unchanged),
            len(report.removed),
            len(report.skipped),
            report.embeddings_computed,
            report.embeddings_from_cache,
        )
        return report
