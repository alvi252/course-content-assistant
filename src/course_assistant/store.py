"""SQLite persistence for documents, chunks, vectors and the embedding cache.

A single file, no server. Vectors are float32 blobs; search happens in NumPy (see
``retrieval.py``). That is plenty for thousands of chunks; for millions, swap this class for
pgvector, Qdrant or similar behind the same methods.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IndexModelMismatchError
from .models import Chunk, Idea, IdeaStatus, NewIdea

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    heading     TEXT NOT NULL,
    text        TEXT NOT NULL,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model     TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
CREATE TABLE IF NOT EXISTS ideas (
    id         INTEGER PRIMARY KEY,
    chunk_id   TEXT NOT NULL,
    doc_path   TEXT NOT NULL,
    doc_title  TEXT NOT NULL,
    heading    TEXT NOT NULL,
    format     TEXT NOT NULL,
    title      TEXT NOT NULL,
    hook       TEXT NOT NULL,
    outline    TEXT NOT NULL,
    quote      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'suggested'
               CHECK (status IN ('suggested', 'used', 'dismissed')),
    script     TEXT,
    caption    TEXT,
    created_at TEXT NOT NULL,
    used_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_ideas_chunk ON ideas(chunk_id);
CREATE INDEX IF NOT EXISTS idx_ideas_doc ON ideas(doc_path);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_iso(moment: datetime | None = None) -> str:
    """UTC timestamp with second precision, e.g. ``2026-10-27T09:30:00+00:00``.

    >>> utc_iso(datetime(2026, 10, 27, 9, 30, tzinfo=UTC))
    '2026-10-27T09:30:00+00:00'
    """
    return (moment or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")


def _idea_from_row(row: sqlite3.Row) -> Idea:
    status: IdeaStatus = row["status"]
    return Idea(
        id=row["id"],
        status=status,
        chunk_id=row["chunk_id"],
        doc_path=row["doc_path"],
        doc_title=row["doc_title"],
        heading=row["heading"],
        format=row["format"],
        title=row["title"],
        hook=row["hook"],
        outline=[str(item) for item in json.loads(row["outline"])],
        quote=row["quote"],
        created_at=row["created_at"],
        used_at=row["used_at"],
        script=row["script"],
        caption=row["caption"],
    )


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.shape[0] != dim:
        raise ValueError(f"Corrupt vector: expected {dim} dims, got {vector.shape[0]}")
    return vector


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection per operation: safe across FastAPI worker threads."""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:  # commits on success, rolls back on exception
                yield conn
        finally:
            conn.close()

    # --- documents -------------------------------------------------------------------

    def fingerprints(self) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT path, fingerprint FROM documents").fetchall()
        return {row["path"]: row["fingerprint"] for row in rows}

    def replace_document(
        self,
        *,
        path: str,
        title: str,
        fingerprint: str,
        model: str,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """Atomically swap a document's chunks: readers see the old or the new version."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute("DELETE FROM documents WHERE path = ?", (path,))
            cursor = conn.execute(
                "INSERT INTO documents (path, title, fingerprint, ingested_at) VALUES (?, ?, ?, ?)",
                (path, title, fingerprint, now),
            )
            document_id = cursor.lastrowid
            conn.executemany(
                "INSERT INTO chunks"
                " (id, document_id, ordinal, heading, text, model, dim, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, document_id, c.ordinal, c.heading, c.text, model, len(v), _to_blob(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_documents(self, paths: Iterable[str]) -> None:
        with self.connect() as conn:
            conn.executemany("DELETE FROM documents WHERE path = ?", [(p,) for p in paths])

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT d.path, d.title, d.ingested_at, COUNT(c.id) AS chunks"
                " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
                " GROUP BY d.id ORDER BY d.title"
            ).fetchall()
        return [dict(row) for row in rows]

    def load_index(self) -> tuple[list[Chunk], np.ndarray, str | None]:
        """Return every chunk, their vectors as one matrix, and the embedding model name."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.ordinal, c.heading, c.text, c.model, c.dim, c.embedding,"
                " d.path, d.title FROM chunks c JOIN documents d ON d.id = c.document_id"
                " ORDER BY d.path, c.ordinal"
            ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32), None
        models = {row["model"] for row in rows}
        if len(models) > 1:
            raise IndexModelMismatchError(
                f"Index mixes embedding models {sorted(models)}. Run ingest --rebuild."
            )
        chunks = [
            Chunk(
                id=row["id"],
                doc_path=row["path"],
                doc_title=row["title"],
                heading=row["heading"],
                ordinal=row["ordinal"],
                text=row["text"],
            )
            for row in rows
        ]
        matrix = np.vstack([_from_blob(row["embedding"], row["dim"]) for row in rows])
        return chunks, matrix, models.pop()

    def load_chunks(self) -> list[Chunk]:
        """Every chunk (text and place only), without vectors and without the model check.

        This keeps listing, coverage and status changes working even while the index is
        half-way through a re-ingest with another embedding model.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.ordinal, c.heading, c.text, d.path, d.title FROM chunks c"
                " JOIN documents d ON d.id = c.document_id ORDER BY d.path, c.ordinal"
            ).fetchall()
        return [
            Chunk(
                id=row["id"],
                doc_path=row["path"],
                doc_title=row["title"],
                heading=row["heading"],
                ordinal=row["ordinal"],
                text=row["text"],
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": int(documents), "chunks": int(chunks)}

    # --- embedding cache ---------------------------------------------------------------

    def cached_embeddings(self, model: str, text_hashes: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self.connect() as conn:
            for start in range(0, len(text_hashes), 500):
                batch = list(text_hashes[start : start + 500])
                placeholders = ",".join("?" * len(batch))
                # Only "?" placeholders are interpolated; values are bound parameters.
                query = (
                    "SELECT text_hash, dim, vector FROM embedding_cache"  # noqa: S608
                    f" WHERE model = ? AND text_hash IN ({placeholders})"
                )
                rows = conn.execute(query, [model, *batch]).fetchall()
                for row in rows:
                    found[row["text_hash"]] = _from_blob(row["vector"], row["dim"])
        return found

    def cache_embeddings(self, model: str, items: dict[str, np.ndarray]) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (model, text_hash, dim, vector)"
                " VALUES (?, ?, ?, ?)",
                [(model, h, len(v), _to_blob(v)) for h, v in items.items()],
            )

    # --- change tracking ---------------------------------------------------------------

    def generation(self) -> int:
        """A counter bumped after every change, so readers know when to reload."""
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()
        return int(row["value"]) if row else 0

    def bump_generation(self) -> int:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('generation', '1')"
                " ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
            )
            row = conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()
        return int(row["value"])

    # --- ideas ---------------------------------------------------------------------------

    def add_ideas(self, ideas: Sequence[NewIdea], *, now: datetime | None = None) -> list[Idea]:
        """Save new ideas as 'suggested' and return them with their ids."""
        created = utc_iso(now)
        saved: list[Idea] = []
        with self.connect() as conn:
            for idea in ideas:
                cursor = conn.execute(
                    "INSERT INTO ideas (chunk_id, doc_path, doc_title, heading, format, title,"
                    " hook, outline, quote, status, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'suggested', ?)",
                    (
                        idea.chunk_id,
                        idea.doc_path,
                        idea.doc_title,
                        idea.heading,
                        idea.format,
                        idea.title,
                        idea.hook,
                        json.dumps(idea.outline, ensure_ascii=False),
                        idea.quote,
                        created,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM ideas WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                saved.append(_idea_from_row(row))
        return saved

    def get_idea(self, idea_id: int) -> Idea | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        return _idea_from_row(row) if row else None

    def list_ideas(self, *, status: IdeaStatus | None = None, limit: int = 100) -> list[Idea]:
        """Newest first, optionally only one status."""
        query = "SELECT * FROM ideas"
        params: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_idea_from_row(row) for row in rows]

    def set_idea_status(
        self, idea_id: int, status: IdeaStatus, *, now: datetime | None = None
    ) -> Idea | None:
        """'used' stamps ``used_at``; any other status clears it. None if the idea is missing."""
        used_at = utc_iso(now) if status == "used" else None
        with self.connect() as conn:
            conn.execute(
                "UPDATE ideas SET status = ?, used_at = ? WHERE id = ?",
                (status, used_at, idea_id),
            )
        return self.get_idea(idea_id)

    def set_idea_script(self, idea_id: int, script: str, caption: str) -> Idea | None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE ideas SET script = ?, caption = ? WHERE id = ?", (script, caption, idea_id)
            )
        return self.get_idea(idea_id)

    def idea_counts(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM ideas GROUP BY status"
            ).fetchall()
        counts = {"suggested": 0, "used": 0, "dismissed": 0}
        counts.update({row["status"]: int(row["n"]) for row in rows})
        return counts

    def rebind_ideas(self, places: Sequence[tuple[int, Chunk]]) -> None:
        """Point ideas at the chunks that hold them now (id, document and section).

        The quote, title, hook, outline, status, script and timestamps are never changed.
        """
        with self.connect() as conn:
            conn.executemany(
                "UPDATE ideas SET chunk_id = ?, doc_path = ?, doc_title = ?, heading = ?"
                " WHERE id = ?",
                [(c.id, c.doc_path, c.doc_title, c.heading, idea_id) for idea_id, c in places],
            )

    def all_ideas(self) -> list[Idea]:
        """The whole log, oldest first (it is small: a few hundred rows at most)."""
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM ideas ORDER BY id").fetchall()
        return [_idea_from_row(row) for row in rows]
