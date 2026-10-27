"""Deterministic checks that flag numbers the model may have added to the author's material.

A number in an idea or a script that is not in the source passage is the most common way a
model drifts from what the author actually teaches ("five dinners" when the lesson says four).
These checks do not delete anything; they produce warnings for the author to look at.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# "one" is left out on purpose: it is far more often a pronoun ("the one thing") than a count.
_WORD_VALUES = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}  # fmt: skip
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*|\b(?:" + "|".join(_WORD_VALUES) + r")\b", re.IGNORECASE)
# Slide / list numbering ("1. ", "2) ", "Slide 3: ") is structure, not a claim. A number that
# starts a sentence ("30-minute dinners", "2.5 cups") is a claim and stays.
_LIST_NUMBER = re.compile(r"^\s*(?:slide\s+\d+\s*[:.)-]|\d+[.)])\s+", re.IGNORECASE | re.MULTILINE)
# Labels like "Slide 2", "Step 3", "Line 1", "Part 2", "Tip 4" number the piece itself.
_SLIDE_WORD = re.compile(r"\b(?:slide|step|line|part|tip)\s+\d+\b", re.IGNORECASE)


def _value(token: str) -> str:
    return _WORD_VALUES.get(token.lower(), token)


def numbers_missing_from_source(text: str, source: str) -> list[str]:
    """Numbers in ``text`` (as digits or words) that ``source`` does not contain in either form.

    Returned as written in ``text``, in order, once each. Slide and list numbering is ignored.

    >>> numbers_missing_from_source("Takes 90 minutes, serves 4. 90 again.", "It takes 4 hours.")
    ['90']
    >>> numbers_missing_from_source("Five different dinners", "Same tray, four dinners.")
    ['Five']
    >>> numbers_missing_from_source("Ninety minutes, 4 dinners", "90-minute Sunday, four dinners")
    []
    >>> numbers_missing_from_source("1. First\\n2. Second\\nSlide 3: Third", "no digits here")
    []
    >>> numbers_missing_from_source("Line 1: base. Line 2: protein. Step 3: sauce", "x")
    []
    >>> numbers_missing_from_source("30-minute dinners", "twenty-minute frame")
    ['30']
    >>> numbers_missing_from_source("2.5 cups of rice", "add 2.5 cups")
    []
    >>> numbers_missing_from_source("The one thing to do", "nothing")
    []
    """
    cleaned = _SLIDE_WORD.sub(" ", _LIST_NUMBER.sub("", text))
    known = {_value(token) for token in _NUMBER.findall(source)}
    missing: list[str] = []
    seen: set[str] = set()
    for token in _NUMBER.findall(cleaned):
        value = _value(token)
        if value not in known and value not in seen:
            missing.append(token)
            seen.add(value)
    return missing


def idea_warnings(title: str, hook: str, outline: Sequence[str], source: str) -> list[str]:
    """Plain-language warnings for an idea whose wording adds numbers to the passage.

    >>> idea_warnings("One tray, five dinners", "Same tray, four dinners.", ["Roast once"],
    ...               "Same tray, four dinners, and none of them feels like leftovers.")
    ['Number five is in the title but not in the source passage.']
    >>> idea_warnings("Three jars", "One creamy, one sharp", [], "make three jars")
    []
    """
    warnings: list[str] = []
    for where, text in (("title", title), ("hook", hook), ("outline", "\n".join(outline))):
        for number in numbers_missing_from_source(text, source):
            warnings.append(
                f"Number {number.lower()} is in the {where} but not in the source passage."
            )
    return warnings
