"""Checks on the model's ideas, done by code: the model's own claims are never trusted.

The key check is the quote: it must be a real, long-enough piece of the passage the idea is
built on, so every idea can be traced back to the author's own words.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from .errors import GenerationError
from .models import FORMATS, Chunk, DroppedIdea, DropReason, NewIdea

MIN_QUOTE_WORDS = 8

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_QUOTE_MARKS = str.maketrans(
    {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'}
)
_DASHES = str.maketrans({"–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-"})
_QUOTE_LEAD = " \t\n\"'([.-"
_QUOTE_TAIL = " \t\n\"').,;:!?-]"


def normalize_text(text: str) -> str:
    """Make two spellings of the same words comparable.

    NFKC, curly quotes to straight, en/em dashes to a hyphen, whitespace collapsed, lowercase.

    >>> normalize_text("Don’t  waste\\nthe “base” — ever.")
    'don\\'t waste the "base" - ever.'
    >>> normalize_text("Wait…")
    'wait...'
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MARKS).translate(_DASHES)
    return " ".join(text.split()).casefold()


def clean_quote(quote: str) -> str:
    """Normalise a quote and trim quote marks, ellipses and closing punctuation at its edges.

    >>> clean_quote("“Cook the rice once, eat it three ways…”")
    'cook the rice once, eat it three ways'
    >>> clean_quote("...and that is the whole trick.")
    'and that is the whole trick'
    """
    return normalize_text(quote).lstrip(_QUOTE_LEAD).rstrip(_QUOTE_TAIL)


def quote_word_count(quote: str) -> int:
    """Words in a cleaned quote.

    >>> quote_word_count("“One two three.”")
    3
    """
    return len(clean_quote(quote).split())


def text_contains_quote(normalized_text: str, cleaned_quote: str) -> bool:
    """Whole-word containment of an already cleaned quote in an already normalised text.

    The quote must start and end on word boundaries: ``"planned night"`` is not found in
    ``"an unplanned night"``, and ``"batch cooking is"`` is not found in ``"batch cooking isn't"``.

    >>> text_contains_quote("an unplanned night is not a failed plan", "planned night is not")
    False
    >>> text_contains_quote("decide batch cooking isn't for them", "decide batch cooking is")
    False
    >>> text_contains_quote("an unplanned night is not a failed plan", "night is not a failed")
    True
    >>> text_contains_quote("anything", "")
    False

    Apostrophes count as part of a word, so a quote cannot stop in front of "n't" and flip the
    meaning:

    >>> lesson = "a freezer full of square tubs is a freezer you can't see into."
    >>> text_contains_quote(lesson, "a freezer full of square tubs is a freezer you can")
    False
    >>> text_contains_quote(lesson, "a freezer full of square tubs is a freezer you can't see into")
    True
    """
    if not cleaned_quote:
        return False
    pattern = rf"(?<!\w)(?<!\w'){re.escape(cleaned_quote)}(?!\w|'\w)"
    return re.search(pattern, normalized_text) is not None


def quote_in_passage(quote: str, passage: str) -> bool:
    """True if the cleaned quote is a verbatim, whole-word piece of the passage.

    >>> p = "Cook the rice once.  Then eat it three ways -- bowls, wraps, soup."
    >>> quote_in_passage("“Then eat it three ways -- bowls, wraps, soup.”", p)
    True
    >>> quote_in_passage("then EAT it three ways", p)
    True
    >>> quote_in_passage("Then eat it four ways", p)
    False
    >>> quote_in_passage("", p)
    False
    >>> q = "planned night is not a failed plan. It is part of the plan."
    >>> quote_in_passage(q, "An unplanned night is not a failed plan. It is part of the plan.")
    False
    >>> q = "By Friday they order takeaway and decide batch cooking is"
    >>> quote_in_passage(q, "By Friday they order takeaway and decide batch cooking isn't for them.")
    False
    """  # noqa: E501
    return text_contains_quote(normalize_text(passage), clean_quote(quote))


def parse_ideas_output(text: str) -> list[Any]:
    """The ``ideas`` list from the model's reply; tolerates prose around the JSON object.

    >>> parse_ideas_output('{"ideas": [{"passage": 1}]}')
    [{'passage': 1}]
    >>> parse_ideas_output('Sure! {"ideas": []} Hope it helps')
    []
    >>> parse_ideas_output("not json")
    Traceback (most recent call last):
    ...
    course_assistant.errors.GenerationError: The model's reply was not valid JSON ...
    """
    for candidate in (text, *(m.group(0) for m in [_JSON_OBJECT.search(text)] if m)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("ideas"), list):
            return data["ideas"]
    raise GenerationError(
        "The model's reply was not valid JSON with an 'ideas' list (it may have been cut off)."
    )


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def validate_ideas(
    raw_items: Sequence[object],
    passages: Sequence[Chunk],
    requested_format: str | None = None,
) -> tuple[list[NewIdea], list[DroppedIdea]]:
    """Keep only the ideas that pass every check; say why the others were dropped.

    Passages are numbered from 1, in the order they were shown to the model.

    >>> p = Chunk("c1", "course/01.md", "Lesson", "Intro", 0, "Cook the rice once and then eat it three ways this week.")
    >>> good = {"passage": 1, "format": "post", "title": "T", "hook": "H", "outline": ["a"],
    ...         "quote": "Cook the rice once and then eat it three ways"}
    >>> kept, dropped = validate_ideas([good, dict(good, title="Again")], [p])
    >>> [i.title for i in kept], [(d.reason, d.title) for d in dropped]
    (['T'], [('duplicate_passage', 'Again')])
    >>> validate_ideas([dict(good, passage=2)], [p])[1][0].reason
    'unknown_passage'
    >>> validate_ideas([good], [p], requested_format="carousel")[1][0].reason
    'wrong_format'
    >>> validate_ideas([dict(good, quote="Cook the rice")], [p])[1][0].reason
    'quote_too_short'
    >>> validate_ideas([dict(good, quote="Bake the rice once and then eat it four ways")], [p])[1][0].reason
    'quote_not_in_source'
    """  # noqa: E501
    kept: list[NewIdea] = []
    dropped: list[DroppedIdea] = []
    taken: set[int] = set()

    for item in raw_items:
        raw = item if isinstance(item, dict) else {}
        title = _text(raw.get("title")) or ""

        def drop(reason: DropReason, title: str = title) -> None:
            dropped.append(DroppedIdea(reason, title))

        number = raw.get("passage")
        fmt = raw.get("format")
        hook = _text(raw.get("hook"))
        quote = _text(raw.get("quote"))
        outline_raw = raw.get("outline")
        if (
            not title
            or not hook
            or not quote
            or not isinstance(number, int)
            or isinstance(number, bool)
            or not isinstance(fmt, str)
            or not isinstance(outline_raw, list)
            or not all(isinstance(step, str) for step in outline_raw)
        ):
            drop("malformed_idea")
            continue
        if not 1 <= number <= len(passages):
            drop("unknown_passage")
            continue
        if fmt not in FORMATS or (requested_format is not None and fmt != requested_format):
            drop("wrong_format")
            continue
        if quote_word_count(quote) < MIN_QUOTE_WORDS:
            drop("quote_too_short")
            continue
        passage = passages[number - 1]
        if not quote_in_passage(quote, passage.text):
            drop("quote_not_in_source")
            continue
        if number in taken:
            drop("duplicate_passage")
            continue
        taken.add(number)
        kept.append(
            NewIdea(
                chunk_id=passage.id,
                doc_path=passage.doc_path,
                doc_title=passage.doc_title,
                heading=passage.heading,
                format=fmt,
                title=title,
                hook=hook,
                outline=[step.strip() for step in outline_raw if step.strip()],
                quote=quote,
            )
        )
    return kept, dropped
