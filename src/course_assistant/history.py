"""What has been done with each passage, seen through the CURRENT index.

The ideas log keeps the chunk id, path and heading as they were when the idea was made. But
chunk ids hash path, position and text, so fixing a typo, adding a paragraph above or renaming
a lesson file gives every affected passage a new id. The quote saved with each idea does not
change, so it is used to find the passage again: an idea counts for the passage (and lesson)
that contains its quote now. Ideas whose quote exists nowhere any more count nowhere.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from .models import Chunk, Idea
from .store import Store, utc_iso
from .validation import clean_quote, normalize_text, text_contains_quote


@dataclass(frozen=True)
class ChunkStats:
    """How often one passage was proposed (any status) and used, and when last used."""

    suggested_count: int = 0
    used_count: int = 0
    last_used_at: str | None = None


@dataclass(frozen=True)
class LessonCoverage:
    doc_path: str
    title: str
    passages: int
    suggested: int  # ideas ever proposed from this lesson
    used: int  # ideas from this lesson marked as used
    used_passages: int  # distinct passages with a used idea
    last_used_at: str | None


class ChunkResolver:
    """Maps an idea to the chunk that holds its quote today."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._by_id = {c.id: c for c in chunks}
        self._normalized: dict[str, str] = {}

    def _text(self, chunk: Chunk) -> str:
        if chunk.id not in self._normalized:
            self._normalized[chunk.id] = normalize_text(chunk.text)
        return self._normalized[chunk.id]

    def resolve(self, idea: Idea) -> Chunk | None:
        """The idea's own chunk if it still exists, else the chunk containing its quote.

        The same document is preferred, then any document. If the quoted sentence itself was
        rewritten, the first chunk of the same document under the same heading is used; if
        that is gone too, the result is ``None`` and the idea counts nowhere.
        """
        own = self._by_id.get(idea.chunk_id)
        if own is not None:
            return own
        quote = clean_quote(idea.quote)
        found = [c for c in self._chunks if text_contains_quote(self._text(c), quote)]
        for chunk in found:
            if chunk.doc_path == idea.doc_path:
                return chunk
        if found:
            return found[0]
        return next(
            (c for c in self._chunks if c.doc_path == idea.doc_path and c.heading == idea.heading),
            None,
        )


class History:
    def __init__(self, ideas: list[Idea], chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.ideas = ideas
        resolver = ChunkResolver(chunks)
        self.resolved: dict[int, Chunk | None] = {i.id: resolver.resolve(i) for i in ideas}

    @classmethod
    def load(cls, store: Store, chunks: list[Chunk]) -> History:
        return cls(store.all_ideas(), chunks)

    def _placed(self) -> list[tuple[Idea, Chunk]]:
        return [(i, c) for i in self.ideas if (c := self.resolved[i.id]) is not None]

    def chunk_stats(self) -> dict[str, ChunkStats]:
        """Per current chunk id. Only chunks with at least one idea appear."""
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        last: dict[str, str] = {}
        for idea, chunk in self._placed():
            counts[chunk.id][0] += 1
            if idea.status == "used":
                counts[chunk.id][1] += 1
                if idea.used_at and idea.used_at > last.get(chunk.id, ""):
                    last[chunk.id] = idea.used_at
        return {cid: ChunkStats(n[0], n[1], last.get(cid)) for cid, n in counts.items()}

    def lesson_used_counts(self) -> dict[str, int]:
        """Used ideas per current document path."""
        counts: dict[str, int] = defaultdict(int)
        for idea, chunk in self._placed():
            if idea.status == "used":
                counts[chunk.doc_path] += 1
        return dict(counts)

    def recently_used_chunk_ids(self, since: datetime) -> set[str]:
        """Current chunks with a 'used' idea stamped at or after ``since`` (inclusive)."""
        cutoff = utc_iso(since)
        return {
            chunk.id
            for idea, chunk in self._placed()
            if idea.status == "used" and idea.used_at is not None and idea.used_at >= cutoff
        }

    def coverage(self) -> list[LessonCoverage]:
        """One row per lesson in the index, sorted by path (course folders group together)."""
        passages: dict[str, int] = defaultdict(int)
        titles: dict[str, str] = {}
        for chunk in self.chunks:
            passages[chunk.doc_path] += 1
            titles.setdefault(chunk.doc_path, chunk.doc_title)
        suggested: dict[str, int] = defaultdict(int)
        used: dict[str, int] = defaultdict(int)
        used_chunks: dict[str, set[str]] = defaultdict(set)
        last: dict[str, str] = {}
        for idea, chunk in self._placed():
            path = chunk.doc_path
            suggested[path] += 1
            if idea.status == "used":
                used[path] += 1
                used_chunks[path].add(chunk.id)
                if idea.used_at and idea.used_at > last.get(path, ""):
                    last[path] = idea.used_at
        return [
            LessonCoverage(
                doc_path=path,
                title=titles[path],
                passages=passages[path],
                suggested=suggested[path],
                used=used[path],
                used_passages=len(used_chunks[path]),
                last_used_at=last.get(path),
            )
            for path in sorted(passages)
        ]


def rebind_ideas(store: Store) -> int:
    """Point every logged idea at the chunk that holds it today; returns how many moved.

    Run after an ingest that changed something, so later requests find the idea's own chunk id
    at once instead of searching for the quote. ``chunk_id``, ``doc_path``, ``doc_title`` and
    ``heading`` are updated to the current chunk's; the quote, title, hook, outline, status,
    script and timestamps stay exactly as they were. Ideas that resolve nowhere are left alone.
    """
    resolver = ChunkResolver(store.load_chunks())
    moves = []
    for idea in store.all_ideas():
        chunk = resolver.resolve(idea)
        if chunk is not None and (chunk.id, chunk.doc_path, chunk.heading) != (
            idea.chunk_id,
            idea.doc_path,
            idea.heading,
        ):
            moves.append((idea.id, chunk))
    store.rebind_ideas(moves)
    return len(moves)
