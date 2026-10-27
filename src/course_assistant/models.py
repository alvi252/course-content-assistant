"""Plain data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .checks import idea_warnings

FORMATS: dict[str, str] = {
    "talking_head": "Talking-head video",
    "carousel": "Carousel",
    "text_graphic": "Text graphic",
    "post": "Educational post",
}

IdeaStatus = Literal["suggested", "used", "dismissed"]

EmptyReason = Literal[
    "empty_index",
    "no_topic_match",
    "all_too_short",
    "topic_matches_used",
    "all_recently_used",
    "model_returned_none",
]

DropReason = Literal[
    "unknown_passage",
    "duplicate_passage",
    "wrong_format",
    "quote_too_short",
    "quote_not_in_source",
    "malformed_idea",
]


def course_name(doc_path: str) -> str:
    """Course display name: the first path segment, title-cased.

    >>> course_name("sunday-batch-cooking/01-one-base.md")
    'Sunday Batch Cooking'
    >>> course_name("webinars/q-and-a.md")
    'Webinars'
    >>> course_name("loose-file.md")
    'Loose File'
    """
    first = doc_path.split("/", 1)[0]
    if "/" not in doc_path:
        first = first.rsplit(".", 1)[0]
    return " ".join(word.capitalize() for word in first.replace("_", "-").split("-") if word)


def format_label(key: str) -> str:
    """Human label of a format key; unknown keys are shown as they are.

    >>> format_label("carousel")
    'Carousel'
    >>> format_label("reel")
    'reel'
    """
    return FORMATS.get(key, key)


@dataclass(frozen=True)
class Section:
    """A piece of a document under one heading path, before size-based splitting."""

    heading: str
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    path: str  # relative to the docs directory, POSIX separators
    title: str
    sections: list[Section]
    content_hash: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_path: str
    doc_title: str
    heading: str
    ordinal: int
    text: str

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: title and heading give the vector context."""
        return f"{self.doc_title} — {self.heading}\n\n{self.text}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    similarity: float  # cosine similarity between query and chunk, -1..1
    score: float  # final ranking score (RRF score in hybrid mode, cosine otherwise)
    vector_rank: int | None = None
    keyword_rank: int | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float | None = 0.0

    def add_cost(self, cost: float | None) -> None:
        if cost is None or self.cost_usd is None:
            self.cost_usd = None
        else:
            self.cost_usd += cost


@dataclass(frozen=True)
class NewIdea:
    """A validated idea, ready to be saved."""

    chunk_id: str
    doc_path: str
    doc_title: str
    heading: str
    format: str
    title: str
    hook: str
    outline: list[str]
    quote: str


@dataclass(frozen=True)
class Idea:
    id: int
    status: IdeaStatus
    chunk_id: str
    doc_path: str
    doc_title: str
    heading: str
    format: str
    title: str
    hook: str
    outline: list[str]
    quote: str
    created_at: str
    used_at: str | None = None
    script: str | None = None
    caption: str | None = None

    def to_dict(
        self, source_text: str | None = None, current: Chunk | None = None
    ) -> dict[str, Any]:
        """The Idea JSON of the API.

        ``current`` is the chunk that holds the quote in today's index (see ``history``); its
        place and text are shown instead of the ones saved with the idea, which may be stale.
        """
        if current is not None:
            source_text = current.text
        chunk_id = current.id if current else self.chunk_id
        doc_path = current.doc_path if current else self.doc_path
        doc_title = current.doc_title if current else self.doc_title
        heading = current.heading if current else self.heading
        return {
            "id": self.id,
            "status": self.status,
            "format": self.format,
            "format_label": format_label(self.format),
            "title": self.title,
            "hook": self.hook,
            "outline": list(self.outline),
            "quote": self.quote,
            "source": {
                "chunk_id": chunk_id,
                "doc_path": doc_path,
                "course": course_name(doc_path),
                "lesson": doc_title,
                "section": heading,
                "text": source_text,
            },
            "created_at": self.created_at,
            "used_at": self.used_at,
            "script": self.script,
            "caption": self.caption,
            "warnings": (
                idea_warnings(self.title, self.hook, self.outline, source_text)
                if source_text
                else []
            ),
        }


@dataclass(frozen=True)
class DroppedIdea:
    reason: DropReason
    title: str


@dataclass
class IdeasResult:
    ideas: list[Idea]
    dropped: list[DroppedIdea]
    requested: int
    candidates: int  # passages offered to the model
    fresh_passages_left: int  # eligible passages in the whole library at request time
    model: str | None
    empty_reason: EmptyReason | None = None  # why no ideas came back (None when some did)
    source_texts: dict[str, str] = field(default_factory=dict)  # chunk_id -> passage text
    latency_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ideas": [i.to_dict(self.source_texts.get(i.chunk_id)) for i in self.ideas],
            "dropped": [asdict(d) for d in self.dropped],
            "requested": self.requested,
            "candidates": self.candidates,
            "fresh_passages_left": self.fresh_passages_left,
            "empty_reason": self.empty_reason,
            "model": self.model,
            "latency_ms": {k: round(v, 1) for k, v in self.latency_ms.items()},
            "usage": asdict(self.usage),
        }


@dataclass
class ScriptResult:
    idea: Idea
    script: str
    caption: str
    warnings: list[str]
    model: str
    source_text: str | None = None
    current: Chunk | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea.to_dict(self.source_text, self.current),
            "script": self.script,
            "caption": self.caption,
            "warnings": list(self.warnings),
            "model": self.model,
            "latency_ms": {k: round(v, 1) for k, v in self.latency_ms.items()},
            "usage": asdict(self.usage),
        }


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_from_cache: int = 0
    embedding_tokens: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["changed"] = self.changed
        return data
