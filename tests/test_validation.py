from __future__ import annotations

from typing import Any

import pytest

from course_assistant.errors import GenerationError
from course_assistant.models import Chunk
from course_assistant.validation import (
    parse_ideas_output,
    quote_in_passage,
    validate_ideas,
)

TEXT = (
    "Cook a big pot of plain rice and roast one tray of vegetables on Sunday. "
    "Then turn the same base into bowls, wraps and soup. "
    "Don’t season anything — not until the day you eat it."
)
PASSAGES = [
    Chunk("c1", "sunday/01.md", "Lesson 1", "Cook the base", 0, TEXT),
    Chunk("c2", "sunday/02.md", "Lesson 2", "Sauces", 0, "Simmer the sauce for forty minutes."),
]
GOOD: dict[str, Any] = {
    "passage": 1,
    "format": "post",
    "title": "One pot",
    "hook": "Cook once.",
    "outline": ["rice", "tray"],
    "quote": "Cook a big pot of plain rice and roast one tray of vegetables on Sunday.",
}


def _reasons(items: list[dict[str, Any]], fmt: str | None = None) -> list[str]:
    return [d.reason for d in validate_ideas(items, PASSAGES, fmt)[1]]


def test_exact_quote_is_kept_and_carries_the_source() -> None:
    kept, dropped = validate_ideas([GOOD], PASSAGES)
    assert dropped == []
    assert (kept[0].chunk_id, kept[0].doc_path, kept[0].heading) == (
        "c1",
        "sunday/01.md",
        "Cook the base",
    )
    assert kept[0].outline == ["rice", "tray"]


@pytest.mark.parametrize(
    "quote",
    [
        "COOK A BIG POT OF PLAIN RICE AND ROAST ONE TRAY",  # case
        "Cook  a big pot of plain\nrice and roast one tray",  # extra spaces and a newline
        "Don't season anything - not until the day you eat it.",  # straight quote, hyphen
        "“Don’t season anything — not until the day you eat it.”",  # curly
        "Then turn the same base into bowls, wraps and soup…",  # trailing ellipsis
        "...then turn the same base into bowls, wraps and soup...",  # both ends
    ],
)
def test_normalised_quotes_are_accepted(quote: str) -> None:
    assert quote_in_passage(quote, TEXT)
    kept, dropped = validate_ideas([dict(GOOD, quote=quote)], PASSAGES)
    assert (len(kept), dropped) == (1, [])


def test_quote_not_in_the_passage_is_dropped() -> None:
    bad = dict(GOOD, quote="Cook a big pot of plain pasta and roast one tray of vegetables")
    assert _reasons([bad]) == ["quote_not_in_source"]


def test_quote_from_another_passage_is_dropped() -> None:
    other = dict(GOOD, quote="Simmer the sauce for forty minutes and then freeze it flat.")
    assert _reasons([other]) == ["quote_not_in_source"]


def test_short_quote_is_dropped_even_when_present() -> None:
    assert _reasons([dict(GOOD, quote="Cook a big pot of plain rice")]) == ["quote_too_short"]


def test_unknown_passage_number() -> None:
    assert _reasons([dict(GOOD, passage=0), dict(GOOD, passage=3)]) == ["unknown_passage"] * 2


def test_later_duplicate_for_a_passage_is_dropped() -> None:
    kept, dropped = validate_ideas([GOOD, dict(GOOD, title="Second")], PASSAGES)
    assert [i.title for i in kept] == ["One pot"]
    assert [(d.reason, d.title) for d in dropped] == [("duplicate_passage", "Second")]


def test_format_must_be_known_and_match_the_request() -> None:
    assert _reasons([dict(GOOD, format="reel")]) == ["wrong_format"]
    assert _reasons([GOOD], fmt="carousel") == ["wrong_format"]
    assert _reasons([GOOD], fmt="post") == []
    assert _reasons([dict(GOOD, format="carousel")], fmt=None) == []


@pytest.mark.parametrize(
    "broken",
    [
        {"passage": "1"},
        {"passage": True},
        {"title": ""},
        {"hook": None},
        {"outline": "rice"},
        {"outline": [1, 2]},
        {"quote": ""},
    ],
)
def test_malformed_items_are_reported_not_lost(broken: dict[str, Any]) -> None:
    kept, dropped = validate_ideas([dict(GOOD, **broken)], PASSAGES)
    assert kept == []
    assert [d.reason for d in dropped] == ["malformed_idea"]


def test_non_object_item_is_malformed() -> None:
    assert _reasons(["nonsense"]) == ["malformed_idea"]  # type: ignore[list-item]


def test_parse_ideas_output_reads_json_with_noise() -> None:
    assert parse_ideas_output('```json\n{"ideas": [{"passage": 1}]}\n```') == [{"passage": 1}]


@pytest.mark.parametrize("text", ["", "[1, 2]", '{"ideas": "none"}', '{"ideas": [1,'])
def test_unusable_replies_raise(text: str) -> None:
    with pytest.raises(GenerationError):
        parse_ideas_output(text)


def test_quote_must_start_and_end_on_word_boundaries() -> None:
    unplanned = "An unplanned night is not a failed plan. It is part of the plan."
    takeaway = "By Friday they order takeaway and decide batch cooking isn't for them."
    passages = [
        Chunk("c1", "a/1.md", "L", "H", 0, unplanned),
        Chunk("c2", "a/2.md", "L", "H", 0, takeaway),
    ]
    starts_mid_word = dict(
        GOOD, passage=1, quote="planned night is not a failed plan. It is part of the plan."
    )
    ends_mid_word = dict(
        GOOD, passage=2, quote="By Friday they order takeaway and decide batch cooking is"
    )
    _, dropped = validate_ideas([starts_mid_word, ends_mid_word], passages)
    assert [d.reason for d in dropped] == ["quote_not_in_source"] * 2
    whole = dict(
        GOOD, passage=1, quote="An unplanned night is not a failed plan. It is part of the plan."
    )
    assert len(validate_ideas([whole], passages)[0]) == 1
