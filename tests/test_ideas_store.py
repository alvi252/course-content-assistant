from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import new_idea
from course_assistant.config import Settings
from course_assistant.services import Services, build_services

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


def _chunk(services: Services, heading: str):  # type: ignore[no-untyped-def]
    return next(c for c in services.retriever.all_chunks() if c.heading == heading)


def test_ideas_are_saved_as_suggested_with_all_fields(services: Services) -> None:
    chunk = _chunk(services, "Herb oil")
    [idea] = services.store.add_ideas([new_idea(chunk, "carousel", "Herb oil")], now=NOW)
    assert idea.id >= 1
    assert idea.status == "suggested"
    assert idea.created_at == "2026-10-27T12:00:00+00:00"
    assert (idea.used_at, idea.script, idea.caption) == (None, None, None)
    assert idea.outline == ["a", "b"]
    assert services.store.get_idea(idea.id) == idea
    assert services.store.get_idea(999) is None


def test_status_transitions_set_and_clear_used_at(services: Services) -> None:
    [idea] = services.store.add_ideas([new_idea(_chunk(services, "Herb oil"))], now=NOW)
    used = services.store.set_idea_status(idea.id, "used", now=NOW + timedelta(hours=1))
    assert used is not None
    assert (used.status, used.used_at) == ("used", "2026-10-27T13:00:00+00:00")
    dismissed = services.store.set_idea_status(idea.id, "dismissed")
    assert dismissed is not None
    assert (dismissed.status, dismissed.used_at) == ("dismissed", None)
    used_again = services.store.set_idea_status(idea.id, "used", now=NOW)
    assert used_again is not None and used_again.used_at is not None
    reset = services.store.set_idea_status(idea.id, "suggested")
    assert reset is not None
    assert (reset.status, reset.used_at) == ("suggested", None)
    assert services.store.set_idea_status(999, "used") is None


def test_list_is_newest_first_filtered_and_limited(services: Services) -> None:
    store = services.store
    chunk = _chunk(services, "Herb oil")
    first = store.add_ideas([new_idea(chunk, title="first")], now=NOW)[0]
    second = store.add_ideas([new_idea(chunk, title="second")], now=NOW + timedelta(minutes=1))[0]
    third = store.add_ideas([new_idea(chunk, title="third")], now=NOW + timedelta(minutes=1))[0]
    store.set_idea_status(first.id, "dismissed")

    assert [i.title for i in store.list_ideas()] == ["third", "second", "first"]
    assert [i.title for i in store.list_ideas(limit=2)] == ["third", "second"]
    assert [i.title for i in store.list_ideas(status="dismissed")] == ["first"]
    assert {i.id for i in store.list_ideas(status="suggested")} == {second.id, third.id}


def test_script_is_saved_on_the_idea(services: Services) -> None:
    [idea] = services.store.add_ideas([new_idea(_chunk(services, "Herb oil"))], now=NOW)
    saved = services.store.set_idea_script(idea.id, "the script", "the caption")
    assert saved is not None
    assert (saved.script, saved.caption) == ("the script", "the caption")
    assert services.store.set_idea_script(999, "x", "y") is None


def test_store_is_fresh_in_a_new_services_object(services: Services, settings: Settings) -> None:
    [idea] = services.store.add_ideas([new_idea(_chunk(services, "Herb oil"))], now=NOW)
    again = build_services(settings, embedding_provider=services.embedder.provider)
    assert again.store.get_idea(idea.id) == idea
