"""The author's voice: style sheet, sample posts, and deterministic checks on a written script."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .checks import numbers_missing_from_source
from .validation import normalize_text

logger = logging.getLogger(__name__)

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")


@dataclass(frozen=True)
class StyleSheet:
    raw: str = ""
    phrases: list[str] = field(default_factory=list)  # "Phrases I use"
    never_say: list[str] = field(default_factory=list)  # "Never say"


@dataclass(frozen=True)
class Sample:
    path: str
    title: str
    format: str
    text: str


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split ``key: value`` front matter from the body (no nested YAML, quotes are stripped).

    >>> parse_front_matter('---\\ntitle: "Lesson 3: Base"\\nformat: post\\n---\\nBody text.')
    ({'title': 'Lesson 3: Base', 'format': 'post'}, 'Body text.')
    >>> parse_front_matter("No front matter")
    ({}, 'No front matter')
    """
    raw = raw.replace("\r\n", "\n").lstrip("﻿")
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw.strip()
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, raw[match.end() :].strip()


def _clean_phrase(item: str) -> str:
    return item.strip().strip("\"'`“”‘’").strip()


def parse_style_sheet(raw: str) -> StyleSheet:
    """Read the two bullet lists the prompts need; the rest of the sheet stays as free text.

    >>> sheet = parse_style_sheet('''# Voice
    ... Warm and direct.
    ... ## Phrases I use
    ... - "Here is the thing"
    ... - Cook once, eat twice
    ... ## Never say
    ... - game changer
    ... ''')
    >>> sheet.phrases, sheet.never_say
    (['Here is the thing', 'Cook once, eat twice'], ['game changer'])
    """
    lists: dict[str, list[str]] = {"phrases": [], "never": []}
    current: str | None = None
    for line in raw.replace("\r\n", "\n").split("\n"):
        heading = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if heading:
            title = normalize_text(heading.group(1))
            if title.startswith("phrases i use"):
                current = "phrases"
            elif title.startswith("never say"):
                current = "never"
            else:
                current = None
            continue
        bullet = _BULLET.match(line)
        if current and bullet:
            phrase = _clean_phrase(bullet.group(1))
            if phrase:
                lists[current].append(phrase)
    return StyleSheet(raw=raw.strip(), phrases=lists["phrases"], never_say=lists["never"])


def load_style(voice_dir: Path) -> StyleSheet:
    """Read ``style.md``; a missing file gives an empty sheet (and a log warning)."""
    path = voice_dir / "style.md"
    try:
        return parse_style_sheet(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("No style sheet at %s; ideas and scripts will not follow her voice", path)
        return StyleSheet()


def load_samples(voice_dir: Path) -> list[Sample]:
    """Past posts from ``samples/*.md`` in file-name order."""
    folder = voice_dir / "samples"
    if not folder.is_dir():
        return []
    samples = []
    for path in sorted(folder.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if not body:
            continue
        title = meta.get("title") or path.stem.replace("-", " ").strip().capitalize()
        samples.append(Sample(path.name, title, meta.get("format", ""), body))
    return samples


def pick_samples(samples: list[Sample], fmt: str, limit: int = 5) -> list[Sample]:
    """Up to ``limit`` samples, those in the same format first.

    >>> s = [Sample("a", "A", "post", "x"), Sample("b", "B", "carousel", "y"),
    ...      Sample("c", "C", "post", "z")]
    >>> [x.path for x in pick_samples(s, "post", limit=2)]
    ['a', 'c']
    >>> [x.path for x in pick_samples(s, "carousel", limit=2)]
    ['b', 'a']
    """
    same = [s for s in samples if s.format == fmt]
    other = [s for s in samples if s.format != fmt]
    return (same + other)[:limit]


def never_say_hits(text: str, never_say: list[str]) -> list[str]:
    """The "never say" phrases found in ``text`` (case-insensitive).

    >>> never_say_hits("This is a Game-Changer, honestly.", ["game-changer", "hack"])
    ['game-changer']
    >>> never_say_hits("Plain words.", ["hack"])
    []
    """
    haystack = normalize_text(text)
    return [p for p in never_say if normalize_text(p) and normalize_text(p) in haystack]


def script_warnings(script: str, caption: str, source: str, never_say: list[str]) -> list[str]:
    """Plain-language warnings for the author to check before she records or posts.

    >>> script_warnings("Ready in 90 minutes. Total game changer.", "Serves 4.",
    ...                 "Ready in 40 minutes.", ["game changer"])
    ['Number 90 is in the script but not in the source passage.', 'Number 4 is in the caption but not in the source passage.', 'The script uses a phrase you never say: "game changer".']
    """  # noqa: E501
    warnings: list[str] = []
    for where, text in (("script", script), ("caption", caption)):
        for number in numbers_missing_from_source(text, source):
            warnings.append(f"Number {number} is in the {where} but not in the source passage.")
    for where, text in (("script", script), ("caption", caption)):
        for phrase in never_say_hits(text, never_say):
            warnings.append(f'The {where} uses a phrase you never say: "{phrase}".')
    return warnings
