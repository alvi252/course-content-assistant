from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import new_idea
from course_assistant.ideas import select_passages
from course_assistant.models import Chunk
from course_assistant.services import Services

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


def _chunk(services: Services, heading: str) -> Chunk:
    return next(c for c in services.retriever.all_chunks() if c.heading == heading)


def _use(services: Services, heading: str, days_ago: float) -> None:
    """Log an idea on this section and mark it used ``days_ago`` days before NOW."""
    [idea] = services.store.add_ideas([new_idea(_chunk(services, heading))], now=NOW)
    services.store.set_idea_status(idea.id, "used", now=NOW - timedelta(days=days_ago))


def _paths(services: Services, **kwargs: object) -> list[str]:
    options: dict[str, object] = {"min_chars": 200, "seed": 1, "now": NOW}
    options.update(kwargs)
    selection = select_passages(services.store, services.retriever, **options)  # type: ignore[arg-type]
    return [c.doc_path for c in selection.passages]


def _select(services: Services, **kwargs: object) -> list[str]:
    options: dict[str, object] = {
        "count": 20,
        "skip_used_days": 30,
        "per_lesson": 2,
        "min_chars": 200,
        "seed": 1,
        "now": NOW,
    }
    options.update(kwargs)
    selection = select_passages(services.store, services.retriever, **options)  # type: ignore[arg-type]
    return [c.heading for c in selection.passages]


def test_short_chunks_are_never_selected(services: Services) -> None:
    headings = _select(services)
    assert "Quick note" not in headings
    assert len(headings) == 8
    assert "Quick note" in _select(services, min_chars=10, per_lesson=3)


def test_recently_used_passages_are_skipped(services: Services) -> None:
    _use(services, "Tomato sauce", days_ago=10)
    assert "Tomato sauce" not in _select(services, skip_used_days=30)
    # Outside a shorter window, and with the exclusion off, it is a candidate again.
    assert "Tomato sauce" in _select(services, skip_used_days=5)
    assert "Tomato sauce" in _select(services, skip_used_days=0)


def test_window_boundary_and_old_use(services: Services) -> None:
    _use(services, "Herb oil", days_ago=45)
    assert "Herb oil" in _select(services, skip_used_days=30)


def test_suggested_or_dismissed_ideas_do_not_exclude(services: Services) -> None:
    store = services.store
    [a, b] = store.add_ideas(
        [new_idea(_chunk(services, "Pan noodles")), new_idea(_chunk(services, "Herb oil"))],
        now=NOW,
    )
    store.set_idea_status(b.id, "dismissed", now=NOW)
    assert a.status == "suggested"
    headings = _select(services)
    assert "Pan noodles" in headings
    assert "Herb oil" in headings


def test_least_used_passages_come_first(services: Services) -> None:
    for heading in ("Cook the base", "Store it safely", "Tomato sauce"):
        _use(services, heading, days_ago=60)  # old enough to be eligible, but used before
    picked = _select(services, count=5, per_lesson=3)
    assert len(picked) == 5
    assert not {"Cook the base", "Store it safely", "Tomato sauce"} & set(picked)


def test_proposed_but_unused_passages_rank_after_untouched_ones(services: Services) -> None:
    services.store.add_ideas([new_idea(_chunk(services, "Pan noodles"))], now=NOW)
    picked = _select(services, count=7, per_lesson=3)
    assert "Pan noodles" not in picked
    assert "Pan noodles" in _select(services, count=8, per_lesson=3)


def test_per_lesson_cap_and_lessons_take_turns(services: Services) -> None:
    all_paths = {c.doc_path for c in services.retriever.all_chunks()}

    one_each = _paths(services, count=20, per_lesson=1)
    assert sorted(one_each) == sorted(all_paths)  # capped at one per lesson: four in total

    two_each = _paths(services, count=20, per_lesson=2)
    assert len(two_each) == 8
    assert all(two_each.count(p) <= 2 for p in all_paths)

    first_four = _paths(services, count=4, per_lesson=2)
    assert len(set(first_four)) == 4  # every lesson gets one before any gets a second


def test_lessons_with_fewer_uses_go_first(services: Services) -> None:
    _use(services, "Cook the base", days_ago=60)  # lesson "One base" has been used once
    assert "sunday-batch-cooking/01-one-base.md" not in _paths(services, count=3, per_lesson=2)


def test_same_seed_gives_same_order_and_seeds_can_differ(services: Services) -> None:
    runs = {seed: _select(services, count=5, seed=seed) for seed in range(8)}
    assert _select(services, count=5, seed=4) == runs[4]
    assert len({tuple(v) for v in runs.values()}) > 1


def test_fresh_passages_left_counts_eligible_before_the_cut(services: Services) -> None:
    _use(services, "Tomato sauce", days_ago=1)
    selection = select_passages(
        services.store,
        services.retriever,
        count=3,
        per_lesson=2,
        min_chars=200,
        seed=1,
        now=NOW,
    )
    assert len(selection.passages) == 3
    assert selection.fresh_passages_left == 7  # 9 chunks - 1 too short - 1 used recently


def test_topic_keeps_search_order_and_applies_exclusion(services: Services) -> None:
    services.retriever.min_similarity = 0.0
    top = select_passages(
        services.store,
        services.retriever,
        count=3,
        topic="breakfast for dinner eggs",
        per_lesson=2,
        min_chars=200,
        now=NOW,
    )
    assert top.passages[0].heading == "Breakfast for dinner"
    assert top.embedding_tokens > 0

    _use(services, "Breakfast for dinner", days_ago=2)
    again = select_passages(
        services.store,
        services.retriever,
        count=3,
        topic="breakfast for dinner eggs",
        per_lesson=2,
        min_chars=200,
        now=NOW,
    )
    assert "Breakfast for dinner" not in [c.heading for c in again.passages]


def test_topic_respects_per_lesson_cap(services: Services) -> None:
    services.retriever.min_similarity = 0.0
    result = select_passages(
        services.store,
        services.retriever,
        count=8,
        topic="dinner",
        per_lesson=1,
        min_chars=200,
        now=NOW,
    )
    paths = [c.doc_path for c in result.passages]
    assert len(paths) == len(set(paths))


def test_topic_with_no_match_gives_nothing(services: Services) -> None:
    result = select_passages(
        services.store,
        services.retriever,
        count=5,
        topic="quantum chromodynamics lattice gluon",
        per_lesson=2,
        min_chars=200,
        now=NOW,
    )
    assert result.passages == []
    assert result.search_hits == 0
    assert result.fresh_passages_left == 8  # the library still has fresh passages


def test_fresh_passages_left_is_the_same_with_and_without_a_topic(services: Services) -> None:
    services.retriever.min_similarity = 0.0
    _use(services, "Tomato sauce", days_ago=1)
    kwargs = {"count": 2, "per_lesson": 2, "min_chars": 200, "seed": 1, "now": NOW}
    plain = select_passages(services.store, services.retriever, **kwargs)  # type: ignore[arg-type]
    topical = select_passages(
        services.store,
        services.retriever,
        topic="breakfast",
        **kwargs,  # type: ignore[arg-type]
    )
    assert plain.fresh_passages_left == topical.fresh_passages_left == 7
    assert len(topical.passages) == 2  # asked for 2, though 7 are fresh


def test_window_edge_is_inclusive(services: Services) -> None:
    """Used exactly ``skip_used_days`` ago is still excluded; one minute older is eligible."""
    _use(services, "Tomato sauce", days_ago=30)
    assert "Tomato sauce" not in _select(services, skip_used_days=30)
    _use(services, "Herb oil", days_ago=30 + 1 / 1440)  # one minute older than the window
    assert "Herb oil" in _select(services, skip_used_days=30)
    _use(services, "Pan noodles", days_ago=30 - 1 / 1440)  # one minute inside the window
    assert "Pan noodles" not in _select(services, skip_used_days=30)
