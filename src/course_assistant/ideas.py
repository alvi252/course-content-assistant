"""Pick passages (least-used first), ask the model for ideas, check them, write scripts."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import Settings
from .errors import GenerationError, IdeaNotFoundError, PassageUnavailableError
from .history import ChunkResolver, History
from .llm import LLMProvider
from .models import (
    FORMATS,
    Chunk,
    EmptyReason,
    Idea,
    IdeasResult,
    IdeaStatus,
    ScriptResult,
    Usage,
)
from .prompts import (
    IDEAS_SCHEMA,
    SCRIPT_SCHEMA,
    build_ideas_system,
    build_ideas_user,
    build_script_system,
    build_script_user,
)
from .retrieval import Retriever
from .store import Store
from .usage import UsageLog, estimate_cost
from .validation import parse_ideas_output, validate_ideas
from .voice import load_samples, load_style, pick_samples, script_warnings

logger = logging.getLogger(__name__)

SEARCH_POOL_FACTOR = 3  # with a topic, look at count * 3 search hits before filtering


@dataclass(frozen=True)
class Selection:
    passages: list[Chunk]
    fresh_passages_left: int  # eligible passages in the whole library, whatever the topic
    embedding_tokens: int = 0
    total_chunks: int = 0  # everything in the index
    long_enough: int = 0  # chunks of at least MIN_PASSAGE_CHARS
    search_hits: int | None = None  # topic searches only: hits above MIN_SIMILARITY


def _cap_per_lesson(ordered: list[Chunk], per_lesson: int, count: int) -> list[Chunk]:
    """Walk ``ordered`` keeping at most ``per_lesson`` from each lesson, up to ``count``.

    >>> mk = lambda i, d: Chunk(str(i), d, d, "h", i, "t")
    >>> picked = _cap_per_lesson([mk(1, "a"), mk(2, "a"), mk(3, "a"), mk(4, "b")], 2, 10)
    >>> [c.id for c in picked]
    ['1', '2', '4']
    """
    taken: dict[str, int] = defaultdict(int)
    picked: list[Chunk] = []
    for chunk in ordered:
        if len(picked) >= count:
            break
        if taken[chunk.doc_path] >= per_lesson:
            continue
        taken[chunk.doc_path] += 1
        picked.append(chunk)
    return picked


def _spread(
    lessons: list[list[Chunk]],
    key: Callable[[Chunk], tuple[int, int]],
    lesson_used: dict[str, int],
    per_lesson: int,
    count: int,
) -> list[Chunk]:
    """Pick ``count`` passages: least-used first, lessons taking turns among equals.

    Each lesson is a list already sorted best-first. Every step takes the head passage with the
    lowest ``(key, picks so far from its lesson, uses of its lesson)``; list order breaks ties.
    A lesson never gives more than ``per_lesson`` passages.

    >>> mk = lambda i, d: Chunk(str(i), d, d, "h", i, "t")
    >>> lessons = [[mk(1, "a"), mk(2, "a"), mk(3, "a")], [mk(4, "b")]]
    >>> [c.id for c in _spread(lessons, lambda c: (0, 0), {}, 2, 10)]
    ['1', '4', '2']
    >>> used = {"1": (1, 0), "2": (1, 0)}  # lesson "a" was used before; "b" is untouched
    >>> [c.id for c in _spread(lessons, lambda c: used.get(c.id, (0, 0)), {}, 2, 10)]
    ['4', '1', '2']
    """
    queues = [list(lesson) for lesson in lessons if lesson]
    picks = [0] * len(queues)
    picked: list[Chunk] = []
    while len(picked) < count:
        open_lessons = [i for i, q in enumerate(queues) if q and picks[i] < per_lesson]
        if not open_lessons:
            break
        best = min(
            open_lessons,
            key=lambda i: (
                *key(queues[i][0]),
                picks[i],
                lesson_used.get(queues[i][0].doc_path, 0),
                i,
            ),
        )
        picked.append(queues[best].pop(0))
        picks[best] += 1
    return picked


def select_passages(
    store: Store,
    retriever: Retriever,
    *,
    count: int,
    topic: str | None = None,
    skip_used_days: int = 30,
    per_lesson: int = 2,
    min_chars: int = 200,
    seed: int | None = None,
    now: datetime | None = None,
) -> Selection:
    """Choose the passages to build ideas on.

    Eligible: long enough, and not used (in an idea marked 'used') within ``skip_used_days``
    (0 turns the exclusion off). The window is inclusive: a passage used exactly
    ``skip_used_days`` days ago is still excluded. Use is judged by where each idea's quote
    lives in the current index (see ``history``), so edited or renamed lessons do not make
    used passages look fresh.

    Without a topic the least-used passages and lessons come first (by times used, then times
    proposed) and lessons take turns among equals, the lesson used less going first; with a
    topic the search ranking is kept. Either way at most ``per_lesson`` passages come from one
    lesson. ``fresh_passages_left`` is the number of eligible passages in the whole library,
    whatever the topic.
    """
    moment = now or datetime.now(UTC)
    all_chunks = retriever.all_chunks()
    history = History.load(store, all_chunks)
    recently_used = (
        history.recently_used_chunk_ids(moment - timedelta(days=skip_used_days))
        if skip_used_days > 0
        else set()
    )

    def eligible(chunk: Chunk) -> bool:
        return len(chunk.text) >= min_chars and chunk.id not in recently_used

    long_enough = sum(1 for c in all_chunks if len(c.text) >= min_chars)
    candidates = [c for c in all_chunks if eligible(c)]
    fresh = len(candidates)

    if topic:
        search = retriever.search(topic, top_k=count * SEARCH_POOL_FACTOR)
        ranked = [hit.chunk for hit in search.chunks if eligible(hit.chunk)]
        return Selection(
            _cap_per_lesson(ranked, per_lesson, count),
            fresh,
            search.embedding_tokens,
            total_chunks=len(all_chunks),
            long_enough=long_enough,
            search_hits=len(search.chunks),
        )

    rng = random.Random(seed)  # noqa: S311 - tie-breaking, not security
    rng.shuffle(candidates)  # a stable sort below keeps this order among equals
    chunk_stats = history.chunk_stats()
    lesson_used = history.lesson_used_counts()

    def chunk_key(chunk: Chunk) -> tuple[int, int]:
        stats = chunk_stats.get(chunk.id)
        return (stats.used_count, stats.suggested_count) if stats else (0, 0)

    candidates.sort(key=chunk_key)
    grouped: dict[str, list[Chunk]] = {}
    for chunk in candidates:
        grouped.setdefault(chunk.doc_path, []).append(chunk)
    lessons = list(grouped.values())
    rng.shuffle(lessons)
    picked = _spread(lessons, chunk_key, lesson_used, per_lesson, count)
    return Selection(picked, fresh, total_chunks=len(all_chunks), long_enough=long_enough)


def ideas_token_budget(cfg: Settings, passages: int) -> int:
    """Output tokens to allow for one ideas request: a fixed part plus a share per passage.

    >>> ideas_token_budget(Settings(ideas_tokens_per_idea=250), 30)
    7900
    """
    return 400 + cfg.ideas_tokens_per_idea * passages


def _why_nothing_selected(selection: Selection, topic: str | None) -> EmptyReason:
    """Why no passage was chosen (only called when the selection is empty).

    Without a topic, a library with long enough passages can only come up empty because the
    day window excluded them, so ``all_recently_used`` never appears with ``skip_used_days=0``.
    """
    if selection.total_chunks == 0:
        return "empty_index"
    if topic and selection.search_hits == 0:
        return "no_topic_match"
    if selection.long_enough == 0:
        return "all_too_short"
    if topic:
        return "topic_matches_used"
    return "all_recently_used"


def _record_usage(usage: Usage, log: UsageLog, kind: str, reply_model: str, i: int, o: int) -> None:
    log.record(kind, reply_model, i, o)
    usage.input_tokens += i
    usage.output_tokens += o
    usage.add_cost(estimate_cost(reply_model, i, o))


class IdeaService:
    """Suggest ideas and write scripts. Everything the model returns is checked by code."""

    def __init__(
        self,
        store: Store,
        retriever: Retriever,
        llm: LLMProvider,
        settings: Settings,
        *,
        usage_log: UsageLog | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.usage_log = usage_log or UsageLog(None)

    @property
    def voice_dir(self) -> Path:
        return self.settings.voice_dir

    def suggest(
        self,
        count: int | None = None,
        *,
        fmt: str | None = None,
        topic: str | None = None,
        skip_used_days: int | None = None,
        now: datetime | None = None,
    ) -> IdeasResult:
        cfg = self.settings
        count = cfg.ideas_default_count if count is None else count
        if not 1 <= count <= cfg.ideas_max_count:
            raise ValueError(f"count must be between 1 and {cfg.ideas_max_count}")
        if fmt is not None and fmt not in FORMATS:
            raise ValueError(f"Unknown format {fmt!r}")
        topic = (topic or "").strip() or None

        started = time.perf_counter()
        selection = select_passages(
            self.store,
            self.retriever,
            count=count,
            topic=topic,
            skip_used_days=cfg.skip_used_days if skip_used_days is None else skip_used_days,
            per_lesson=cfg.ideas_per_lesson,
            min_chars=cfg.min_passage_chars,
            seed=cfg.ideas_seed,
            now=now,
        )
        selected_at = time.perf_counter()
        usage = Usage(embedding_tokens=selection.embedding_tokens)
        if selection.embedding_tokens:  # no topic search, no embedding cost
            usage.add_cost(estimate_cost(self.retriever.embedder.model, selection.embedding_tokens))
        result = IdeasResult(
            ideas=[],
            dropped=[],
            requested=count,
            candidates=len(selection.passages),
            fresh_passages_left=selection.fresh_passages_left,
            model=None,
            usage=usage,
        )
        if not selection.passages:
            result.empty_reason = _why_nothing_selected(selection, topic)
            result.latency_ms = _timings(started, selected_at, selected_at)
            return result  # nothing to build on: no LLM call, no cost

        style = load_style(self.voice_dir)
        reply = self.llm.complete(
            system=build_ideas_system(cfg.author_name, style, fmt),
            user=build_ideas_user(selection.passages, topic),
            max_tokens=ideas_token_budget(cfg, len(selection.passages)),
            temperature=cfg.ideas_temperature,
            schema=IDEAS_SCHEMA,
        )
        generated_at = time.perf_counter()
        _record_usage(
            usage, self.usage_log, "ideas", reply.model, reply.input_tokens, reply.output_tokens
        )

        kept, dropped = validate_ideas(parse_ideas_output(reply.text), selection.passages, fmt)
        result.ideas = self.store.add_ideas(kept, now=now)
        result.dropped = dropped
        result.model = reply.model
        result.source_texts = {c.id: c.text for c in selection.passages}
        if not result.ideas:
            result.empty_reason = "model_returned_none"
        result.latency_ms = _timings(started, selected_at, generated_at)
        logger.info(
            "ideas requested=%d candidates=%d kept=%d dropped=%d tokens=%d/%d latency_ms=%.0f",
            count,
            len(selection.passages),
            len(result.ideas),
            len(dropped),
            usage.input_tokens,
            usage.output_tokens,
            result.latency_ms["total"],
        )
        return result

    def write_script(self, idea_id: int) -> ScriptResult:
        idea = self.store.get_idea(idea_id)
        if idea is None:
            raise IdeaNotFoundError(f"No idea with id {idea_id}")
        current = ChunkResolver(self.retriever.all_chunks()).resolve(idea)
        if current is None:
            raise PassageUnavailableError(
                "The passage behind this idea was rewritten or removed; "
                "a script cannot be checked against it."
            )
        passage = current.text
        started = time.perf_counter()
        style = load_style(self.voice_dir)
        samples = pick_samples(load_samples(self.voice_dir), idea.format)
        reply = self.llm.complete(
            system=build_script_system(self.settings.author_name, style, samples, idea.format),
            user=build_script_user(idea, passage),
            max_tokens=self.settings.script_max_tokens,
            temperature=self.settings.script_temperature,
            schema=SCRIPT_SCHEMA,
        )
        generated_at = time.perf_counter()
        usage = Usage()
        _record_usage(
            usage, self.usage_log, "script", reply.model, reply.input_tokens, reply.output_tokens
        )
        script, caption = _parse_script(reply.text)
        saved = self.store.set_idea_script(idea.id, script, caption)
        if saved is None:  # deleted between read and write; cannot happen with this app
            raise IdeaNotFoundError(f"No idea with id {idea_id}")
        return ScriptResult(
            idea=saved,
            script=script,
            caption=caption,
            warnings=script_warnings(script, caption, passage, style.never_say),
            model=reply.model,
            source_text=passage,
            current=current,
            latency_ms=_timings(started, started, generated_at),
            usage=usage,
        )

    def set_status(self, idea_id: int, status: IdeaStatus) -> Idea:
        idea = self.store.set_idea_status(idea_id, status)
        if idea is None:
            raise IdeaNotFoundError(f"No idea with id {idea_id}")
        return idea


def _timings(started: float, selected_at: float, generated_at: float) -> dict[str, float]:
    return {
        "selection": (selected_at - started) * 1000,
        "llm": (generated_at - selected_at) * 1000,
        "total": (time.perf_counter() - started) * 1000,
    }


def _parse_script(text: str) -> tuple[str, str]:
    """``(script, caption)`` from the model's JSON reply.

    >>> _parse_script('{"script": " Hello ", "caption": "Cap"}')
    ('Hello', 'Cap')
    >>> _parse_script('{"script": "", "caption": "Cap"}')
    Traceback (most recent call last):
    ...
    course_assistant.errors.GenerationError: The model's reply had no usable script.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    script = data.get("script") if isinstance(data, dict) else None
    caption = data.get("caption") if isinstance(data, dict) else None
    if not isinstance(script, str) or not script.strip() or not isinstance(caption, str):
        raise GenerationError("The model's reply had no usable script.")
    return script.strip(), caption.strip()
