"""A live check of the idea pipeline, written as Markdown and JSON.

It runs on a COPY of the index, with the ideas log emptied, so the real log is never touched:

1. ``rounds`` times: ask for the default number of ideas (any format) and mark every kept
   idea as used. Over the rounds the assistant must move to fresh material and never repeat.
2. One topic request ("leftovers", 5 ideas).
3. Scripts for the first 3 kept ideas of round 1, with the deterministic warnings.

Run it with ``course-assistant eval``. It calls the real model, so results vary between runs
(the model samples at a temperature above zero); re-run it to see the spread.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .embeddings import EmbeddingProvider
from .llm import LLMProvider
from .models import Idea, IdeasResult, Usage
from .services import Services, build_services
from .validation import normalize_text

TOPIC = "leftovers"
TOPIC_COUNT = 5
SCRIPTS = 3


@dataclass
class RoundReport:
    round: int
    proposed: int  # ideas the model returned (kept + dropped)
    kept: int
    dropped: dict[str, int]
    grounded_rate: float  # kept / proposed
    distinct_lessons: int
    from_new_lessons: int  # kept ideas from lessons with no used idea before this round
    repeats: int  # kept ideas whose passage was already used in an earlier round (must be 0)
    fresh_passages_left: int
    latency_ms: float


@dataclass
class TopicReport:
    topic: str
    requested: int
    passages: int  # passages the search found and sent to the model
    kept: int
    dropped: dict[str, int]
    mention_topic: int  # kept ideas whose passage contains the topic term
    latency_ms: float


@dataclass
class ScriptReport:
    idea_id: int
    title: str
    format: str
    words: int
    warnings: list[str]
    latency_ms: float


@dataclass
class EvalReport:
    generated_at: str
    llm_model: str
    embedding_model: str
    settings: dict[str, Any]
    rounds: list[RoundReport] = field(default_factory=list)
    topic: TopicReport | None = None
    scripts: list[ScriptReport] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mentions_topic(text: str, topic: str) -> bool:
    """Does ``text`` contain the topic term as whole words (case-insensitive, plural folded)?

    A trailing "s" is dropped from the term only when the term is longer than 3 letters and
    does not end in "ss"; the text may then carry an optional plural "s".

    >>> mentions_topic("Breakfast for dinner is the fastest rescue.", "breakfast")
    True
    >>> mentions_topic("Two breakfasts a week.", "Breakfasts")
    True
    >>> mentions_topic("Leftover rice and leftovers.", "leftovers")
    True
    >>> mentions_topic("Leftover rice.", "leftovers")
    True
    >>> mentions_topic("Leftovers again.", "leftover")
    True
    >>> mentions_topic("Unbreakfasted people.", "breakfast")
    False
    >>> mentions_topic("Brunch only.", "breakfast")
    False
    >>> mentions_topic("It is me.", "mess")
    False
    >>> mentions_topic("What a mess.", "mess")
    True
    """
    term = normalize_text(topic)
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        term = term[:-1]
    if not term:
        return False
    return re.search(rf"(?<!\w){re.escape(term)}s?(?!\w)", normalize_text(text)) is not None


def _dropped_by_reason(result: IdeasResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.dropped:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return dict(sorted(counts.items()))


def copy_index(source: Path, target: Path) -> None:
    """Copy a SQLite index with the backup API (safe while it is in WAL mode), minus the log."""
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
        dst.execute("DELETE FROM ideas")
        dst.commit()
    finally:
        src.close()
        dst.close()


def run_eval(
    services: Services,
    *,
    rounds: int = 3,
    topic: str = TOPIC,
    topic_count: int = TOPIC_COUNT,
    script_count: int = SCRIPTS,
) -> EvalReport:
    """Run the scenario on ``services`` (which must already work on a throw-away index)."""
    cfg = services.settings
    store, ideas = services.store, services.ideas
    report = EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        llm_model=ideas.llm.model,
        embedding_model=services.embedder.model,
        settings={
            "count_per_round": cfg.ideas_default_count,
            "ideas_per_lesson": cfg.ideas_per_lesson,
            "skip_used_days": cfg.skip_used_days,
            "min_passage_chars": cfg.min_passage_chars,
            "seed": cfg.ideas_seed,
        },
    )
    usage = Usage()
    llm_latencies: list[float] = []
    llm_calls = 0

    def track(result_usage: Usage, latency_ms: float) -> None:
        nonlocal llm_calls
        llm_calls += 1
        llm_latencies.append(latency_ms)
        usage.input_tokens += result_usage.input_tokens
        usage.output_tokens += result_usage.output_tokens
        usage.embedding_tokens += result_usage.embedding_tokens
        usage.add_cost(result_usage.cost_usd)

    used_chunks: set[str] = set()
    used_lessons: set[str] = set()
    first_round_kept: list[Idea] = []
    for number in range(1, rounds + 1):
        result = ideas.suggest(cfg.ideas_default_count)
        kept = result.ideas
        proposed = len(kept) + len(result.dropped)
        report.rounds.append(
            RoundReport(
                round=number,
                proposed=proposed,
                kept=len(kept),
                dropped=_dropped_by_reason(result),
                grounded_rate=round(len(kept) / proposed, 3) if proposed else 0.0,
                distinct_lessons=len({i.doc_path for i in kept}),
                from_new_lessons=sum(1 for i in kept if i.doc_path not in used_lessons),
                repeats=sum(1 for i in kept if i.chunk_id in used_chunks),
                fresh_passages_left=result.fresh_passages_left,
                latency_ms=round(result.latency_ms.get("total", 0.0), 1),
            )
        )
        if result.model is not None:
            track(result.usage, result.latency_ms.get("llm", 0.0))
        for idea in kept:
            store.set_idea_status(idea.id, "used")
            used_chunks.add(idea.chunk_id)
            used_lessons.add(idea.doc_path)
        if number == 1:
            first_round_kept = list(kept)

    # skip_used_days=0: this request measures search relevance, not the exclusion window.
    result = ideas.suggest(topic_count, topic=topic, skip_used_days=0)
    report.topic = TopicReport(
        topic=topic,
        requested=topic_count,
        passages=result.candidates,
        kept=len(result.ideas),
        dropped=_dropped_by_reason(result),
        mention_topic=sum(
            1
            for i in result.ideas
            if mentions_topic(result.source_texts.get(i.chunk_id, ""), topic)
        ),
        latency_ms=round(result.latency_ms.get("total", 0.0), 1),
    )
    if result.model is not None:
        track(result.usage, result.latency_ms.get("llm", 0.0))

    for idea in first_round_kept[:script_count]:
        scripted = ideas.write_script(idea.id)
        report.scripts.append(
            ScriptReport(
                idea_id=idea.id,
                title=idea.title,
                format=idea.format,
                words=len(scripted.script.split()),
                warnings=scripted.warnings,
                latency_ms=round(scripted.latency_ms.get("total", 0.0), 1),
            )
        )
        track(scripted.usage, scripted.latency_ms.get("llm", 0.0))

    report.totals = {
        "llm_calls": llm_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "embedding_tokens": usage.embedding_tokens,
        "cost_usd": usage.cost_usd,
        "latency_ms_per_call_mean": round(statistics.fmean(llm_latencies), 1)
        if llm_latencies
        else 0.0,
        "latency_ms_per_call_max": round(max(llm_latencies), 1) if llm_latencies else 0.0,
    }
    return report


def run_on_copy(
    settings: Settings,
    *,
    rounds: int = 3,
    embedding_provider: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
) -> EvalReport:
    """Copy the real index to a temp file, run the scenario there, delete the copy."""
    if not settings.index_path.is_file():
        raise FileNotFoundError(f"No index at {settings.index_path}: run `course-assistant ingest`")
    with tempfile.TemporaryDirectory(prefix="course-eval-") as tmp:
        copy = Path(tmp) / "index.sqlite"
        copy_index(settings.index_path, copy)
        services = build_services(
            replace(settings, index_path=copy), embedding_provider=embedding_provider, llm=llm
        )
        if services.store.stats()["chunks"] == 0:
            raise FileNotFoundError("The index is empty: run `course-assistant ingest` first")
        return run_eval(services, rounds=rounds)


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_markdown(report: EvalReport) -> str:
    cfg = report.settings
    lines = [
        "# Idea pipeline check",
        "",
        f"Run at {report.generated_at}. Model `{report.llm_model}`, embeddings "
        f"`{report.embedding_model}`. Run on a copy of the index with an empty ideas log.",
        "",
        f"Each round asks for {cfg['count_per_round']} ideas (any format, at most "
        f"{cfg['ideas_per_lesson']} per lesson, passages used within {cfg['skip_used_days']} days "
        "skipped) and then marks every kept idea as used.",
        "",
        "## Rounds",
        "",
        "| Round | Proposed | Kept | Grounded | Lessons | From new lessons | Repeats | "
        "Dropped | Fresh at start |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rounds:
        dropped = ", ".join(f"{k}: {v}" for k, v in r.dropped.items()) or "none"
        lines.append(
            f"| {r.round} | {r.proposed} | {r.kept} | {_pct(r.grounded_rate)} | "
            f"{r.distinct_lessons} | {r.from_new_lessons} | {r.repeats} | {dropped} | "
            f"{r.fresh_passages_left} |"
        )
    lines += [
        "",
        "Grounded = kept / proposed: the share of ideas whose quote was found in the passage by "
        "code. Repeats must be 0.",
    ]
    if report.topic:
        t = report.topic
        lines += [
            "",
            f'## Topic request: "{t.topic}"',
            "",
            f"Asked for {t.requested}; the search found {t.passages} passages; kept {t.kept}. "
            f"{t.mention_topic} of the kept ideas come "
            f"from passages that mention '{t.topic}'. "
            f"Dropped: {', '.join(f'{k}: {v}' for k, v in t.dropped.items()) or 'none'}.",
        ]
    if report.scripts:
        lines += [
            "",
            "## Scripts (first ideas of round 1)",
            "",
            "| Idea | Format | Words | Warnings |",
            "|---|---|---|---|",
        ]
        for s in report.scripts:
            warnings = "<br>".join(s.warnings) if s.warnings else "none"
            lines.append(f"| {s.title} | {s.format} | {s.words} | {warnings} |")
    t_ = report.totals
    cost = f"${t_['cost_usd']:.4f}" if t_.get("cost_usd") is not None else "n/a"
    lines += [
        "",
        "## Totals",
        "",
        f"- LLM calls: {t_['llm_calls']}",
        f"- Tokens: {t_['input_tokens']} in, {t_['output_tokens']} out, "
        f"{t_['embedding_tokens']} embedding",
        f"- Estimated cost: {cost}",
        f"- Latency per LLM call: mean {t_['latency_ms_per_call_mean']:.0f} ms, "
        f"max {t_['latency_ms_per_call_max']:.0f} ms",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / "report.md", out_dir / "report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return md_path, json_path
