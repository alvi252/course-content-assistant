"""Ideas are tied to passages through their quote, so edits and renames keep the history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from conftest import new_idea
from course_assistant.config import Settings
from course_assistant.errors import PassageUnavailableError
from course_assistant.history import ChunkResolver, History
from course_assistant.models import Chunk, Idea
from course_assistant.services import Services

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
SAUCES = "sunday-batch-cooking/02-sauces.md"


def _chunk(services: Services, heading: str) -> Chunk:
    return next(c for c in services.retriever.all_chunks() if c.heading == heading)


def _history(services: Services) -> History:
    return History.load(services.store, services.retriever.all_chunks())


def _use_herb_oil(services: Services, days_ago: float = 3) -> Idea:
    [idea] = services.store.add_ideas([new_idea(_chunk(services, "Herb oil"))], now=NOW)
    used = services.store.set_idea_status(idea.id, "used", now=NOW - timedelta(days=days_ago))
    assert used is not None
    return used


def test_chunk_stats_count_proposals_and_uses(services: Services) -> None:
    store = services.store
    chunk = _chunk(services, "Herb oil")
    a, b, c = (store.add_ideas([new_idea(chunk)], now=NOW)[0] for _ in range(3))
    store.set_idea_status(a.id, "used", now=NOW)
    store.set_idea_status(b.id, "used", now=NOW + timedelta(days=1))
    store.set_idea_status(c.id, "dismissed")
    history = _history(services)
    stats = history.chunk_stats()[chunk.id]
    assert (stats.suggested_count, stats.used_count) == (3, 2)
    assert stats.last_used_at == "2026-10-28T12:00:00+00:00"
    assert history.lesson_used_counts() == {chunk.doc_path: 2}
    assert store.idea_counts() == {"suggested": 0, "used": 2, "dismissed": 1}


def test_recently_used_window_is_inclusive(services: Services) -> None:
    chunk = _chunk(services, "Herb oil")
    _use_herb_oil(services, days_ago=0)
    history = _history(services)
    assert history.recently_used_chunk_ids(NOW) == {chunk.id}
    assert history.recently_used_chunk_ids(NOW + timedelta(seconds=1)) == set()


def test_coverage_numbers(services: Services) -> None:
    store = services.store
    base, herb = _chunk(services, "Cook the base"), _chunk(services, "Herb oil")
    ideas = store.add_ideas([new_idea(base), new_idea(base), new_idea(herb)], now=NOW)
    store.set_idea_status(ideas[0].id, "used", now=NOW)
    store.set_idea_status(ideas[1].id, "used", now=NOW + timedelta(days=2))

    rows = {r.doc_path: r for r in _history(services).coverage()}

    assert list(rows) == sorted(rows)
    assert len(rows) == 4
    one_base = rows[base.doc_path]
    assert (one_base.passages, one_base.suggested, one_base.used) == (3, 2, 2)
    assert one_base.used_passages == 1  # two used ideas, one passage
    assert one_base.last_used_at == "2026-10-29T12:00:00+00:00"
    sauces = rows[herb.doc_path]
    assert (sauces.suggested, sauces.used, sauces.last_used_at) == (1, 0, None)
    untouched = rows["weeknight-rescue-meals/02-fifteen-minutes.md"]
    assert (untouched.suggested, untouched.used, untouched.used_passages) == (0, 0, 0)


def test_resolver_prefers_own_chunk_then_same_document_then_any(services: Services) -> None:
    chunks = services.retriever.all_chunks()
    herb = _chunk(services, "Herb oil")
    idea = _use_herb_oil(services)
    assert ChunkResolver(chunks).resolve(idea) == herb

    stale = Idea(**{**idea.__dict__, "chunk_id": "gone"})
    assert ChunkResolver(chunks).resolve(stale) == herb
    elsewhere = Idea(**{**idea.__dict__, "chunk_id": "gone", "doc_path": "other/path.md"})
    assert ChunkResolver(chunks).resolve(elsewhere) == herb  # found in another document


def test_same_document_wins_when_the_quote_is_in_two_documents(services: Services) -> None:
    idea = _use_herb_oil(services)
    text = _chunk(services, "Herb oil").text
    first = Chunk("a-id", "a/first.md", "First", "Herb oil", 0, text)
    second = Chunk("b-id", "b/second.md", "Second", "Herb oil", 0, text)
    stale = Idea(**{**idea.__dict__, "chunk_id": "gone", "doc_path": "b/second.md"})
    assert ChunkResolver([first, second]).resolve(stale) == second  # not the first match
    other = Idea(**{**idea.__dict__, "chunk_id": "gone", "doc_path": "c/third.md"})
    assert ChunkResolver([first, second]).resolve(other) == first  # no same-document match


def test_rewritten_quote_falls_back_to_the_same_section(services: Services) -> None:
    chunks = services.retriever.all_chunks()
    herb = _chunk(services, "Herb oil")
    idea = _use_herb_oil(services)
    rewritten = Idea(**{**idea.__dict__, "chunk_id": "gone", "quote": "A sentence nobody wrote"})
    assert ChunkResolver(chunks).resolve(rewritten) == herb  # same document and heading
    moved = Idea(**{**rewritten.__dict__, "heading": "Renamed section"})
    assert ChunkResolver(chunks).resolve(moved) is None  # heading gone too: counts nowhere
    elsewhere = Idea(**{**rewritten.__dict__, "doc_path": "other/path.md"})
    assert ChunkResolver(chunks).resolve(elsewhere) is None  # a heading alone is not enough


def test_quote_must_match_whole_words(services: Services) -> None:
    herb = _chunk(services, "Herb oil")
    idea = _use_herb_oil(services)
    partial = Idea(
        **{
            **idea.__dict__,
            "chunk_id": "gone",
            "doc_path": "other/path.md",
            "quote": "lend a handful of soft herbs with olive",
        }
    )
    assert ChunkResolver([herb]).resolve(partial) is None  # "lend" is the tail of "Blend"


def _edit_above_quote(settings: Settings) -> None:
    path = settings.docs_dir / SAUCES
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("## Herb oil\n\n", "## Herb oil\n\nThis one is the easiest sauce of all. "),
        encoding="utf-8",
    )


def test_editing_the_lesson_keeps_the_passage_used(services: Services, settings: Settings) -> None:
    idea = _use_herb_oil(services)
    old_id = _chunk(services, "Herb oil").id

    _edit_above_quote(settings)
    services.ingestor.run()

    new = _chunk(services, "Herb oil")
    assert new.id != old_id  # the chunk id changed with the text
    history = _history(services)
    assert history.resolved[idea.id] == new
    assert new.id in history.recently_used_chunk_ids(NOW - timedelta(days=30))
    row = next(r for r in history.coverage() if r.doc_path == SAUCES)
    assert (row.used, row.used_passages) == (1, 1)

    from course_assistant.ideas import select_passages

    picked = select_passages(
        services.store,
        services.retriever,
        count=20,
        skip_used_days=30,
        per_lesson=2,
        min_chars=200,
        now=NOW,
    )
    assert "Herb oil" not in [c.heading for c in picked.passages]
    assert picked.fresh_passages_left == 7


def test_script_still_works_after_an_edit(services: Services, settings: Settings) -> None:
    idea = _use_herb_oil(services)
    _edit_above_quote(settings)
    services.ingestor.run()

    result = services.ideas.write_script(idea.id)

    assert result.current is not None and result.current.heading == "Herb oil"
    assert result.to_dict()["idea"]["source"]["text"].startswith("This one is the easiest")


def test_renaming_the_file_keeps_the_history(services: Services, settings: Settings) -> None:
    idea = _use_herb_oil(services)
    renamed = "sunday-batch-cooking/02-sauces-and-oils.md"
    (settings.docs_dir / SAUCES).rename(settings.docs_dir / renamed)

    report = services.ingestor.run()

    assert report.removed == [SAUCES] and report.added == [renamed]
    history = _history(services)
    resolved = history.resolved[idea.id]
    assert resolved is not None and resolved.doc_path == renamed
    rows = {r.doc_path: r for r in history.coverage()}
    assert rows[renamed].used == 1 and SAUCES not in rows
    assert services.ideas.write_script(idea.id).current == resolved


def test_idea_whose_sentence_and_section_are_gone_counts_nowhere(
    services: Services, settings: Settings
) -> None:
    idea = _use_herb_oil(services)
    path = settings.docs_dir / SAUCES
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("Blend a handful", "Whisk a pinch")
        .replace("## Herb oil", "## Green oil"),
        encoding="utf-8",
    )
    services.ingestor.run()

    history = _history(services)
    assert history.resolved[idea.id] is None
    assert all(r.used == 0 and r.suggested == 0 for r in history.coverage())
    with pytest.raises(PassageUnavailableError):
        services.ideas.write_script(idea.id)


def test_rewritten_sentence_in_the_same_section_still_counts(
    services: Services, settings: Settings
) -> None:
    idea = _use_herb_oil(services)
    path = settings.docs_dir / SAUCES
    path.write_text(
        path.read_text(encoding="utf-8").replace("Blend a handful", "Whisk a pinch"),
        encoding="utf-8",
    )
    services.ingestor.run()
    resolved = _history(services).resolved[idea.id]
    assert resolved is not None and resolved.heading == "Herb oil"


def test_ingest_rebinds_the_log_to_the_new_chunk_ids(
    services: Services, settings: Settings
) -> None:
    idea = _use_herb_oil(services)
    _edit_above_quote(settings)
    services.ingestor.run()

    new = _chunk(services, "Herb oil")
    row = services.store.get_idea(idea.id)
    assert row is not None
    assert row.chunk_id == new.id  # later requests take the fast "own chunk id" path
    assert (row.doc_path, row.doc_title, row.heading) == (new.doc_path, new.doc_title, "Herb oil")
    # Nothing else about the idea changed.
    assert (row.quote, row.title, row.status, row.used_at, row.created_at) == (
        idea.quote,
        idea.title,
        idea.status,
        idea.used_at,
        idea.created_at,
    )


def test_rename_rebinds_the_document_path(services: Services, settings: Settings) -> None:
    idea = _use_herb_oil(services)
    renamed = "sunday-batch-cooking/02-sauces-and-oils.md"
    (settings.docs_dir / SAUCES).rename(settings.docs_dir / renamed)
    services.ingestor.run()
    row = services.store.get_idea(idea.id)
    assert row is not None and row.doc_path == renamed
