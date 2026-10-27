"""Shared fixtures: a tiny knowledge base and fake providers. No test touches the network."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from course_assistant.config import Settings
from course_assistant.embeddings import EmbeddingResult, HashingEmbeddings
from course_assistant.llm import LLMResponse
from course_assistant.models import Chunk, NewIdea
from course_assistant.services import Services, build_services

BASE_MD = """\
---
title: "Lesson 1: One base, five meals"
---

## Cook the base

Cook a big pot of plain rice and roast one tray of vegetables on Sunday afternoon. \
Then you can turn the same base into bowls, wraps and soup during the week. \
The trick is to keep everything unseasoned until the day you eat it.

## Store it safely

Cool the base quickly and pack it into shallow containers before it goes in the fridge. \
Label each container with the day you cooked it, and eat everything within four days. \
Don\u2019t leave the pot standing on the stove while you clean up the kitchen.

## Quick note

Keep the lids handy.
"""

SAUCES_MD = """\
# Lesson 2: Sauces that travel

## Tomato sauce

One big batch of tomato sauce becomes pasta night, shakshuka and a pizza base. \
Simmer it for forty minutes with a splash of olive oil and a pinch of salt. \
Freeze it flat in bags so it thaws in minutes when the evening gets busy.

## Herb oil

Blend a handful of soft herbs with olive oil and keep the jar in the fridge. \
A spoonful lifts plain eggs, roasted vegetables and tired leftover rice. \
It keeps well for about a week if you always use a clean spoon.
"""

FRIDGE_MD = """\
---
title: "Lesson 1: The fridge raid"
---

## Breakfast for dinner

Breakfast for dinner is the fastest rescue meal I know when the fridge looks empty. \
Scramble eggs with whatever vegetables are left and serve them on toasted bread. \
Add a spoonful of herb oil if you have it and dinner is done in ten minutes.

## Rescue the leftovers

Look at what is left before you shop and plan one meal around the oldest item. \
Leftover rice becomes fried rice and leftover roast vegetables become a frittata. \
Waste less food and you will spend less at the shop every single week.
"""

FIFTEEN_MD = """\
---
title: "Lesson 2: Fifteen minute dinners"
---

## Pan noodles

Boil noodles while you fry garlic, greens and a spoonful of chilli in a wide pan. \
Toss everything together with a ladle of the noodle water so the sauce clings. \
Dinner is on the table in fifteen minutes and you only wash two pots.

## Sheet pan dinner

Put chopped vegetables and sausages on one tray and roast them at high heat. \
Turn everything once halfway through so the edges brown evenly on both sides. \
Nothing to stir and nothing to watch, so you can help with homework meanwhile.
"""

STYLE_MD = """\
# Mira's voice

Warm, practical and a little dry. Short sentences. She talks to one tired person at the end \
of a long day, never to a crowd.

## Phrases I use

- "Cook once, eat twice"
- Here is the thing
- Keep it boring, keep it easy

## Never say

- game changer
- guilt-free
- clean eating
"""

SAMPLES = {
    "01-sunday-reset.md": (
        "Sunday reset",
        "post",
        "Here is the thing about Sunday. Cook once, eat twice, and Monday gets easier.",
    ),
    "02-rice-slides.md": (
        "Rice slides",
        "carousel",
        "1. Cook the rice\n2. Cool it fast\n3. Eat it all week",
    ),
    "03-fridge-talk.md": (
        "Fridge talk",
        "talking_head",
        "Open the fridge. Keep it boring, keep it easy. [on-screen text: one tray]",
    ),
}


def write_docs(directory: Path) -> Path:
    """Two courses with two lessons each (one section is too short to be a passage)."""
    files = {
        "sunday-batch-cooking/01-one-base.md": BASE_MD,
        "sunday-batch-cooking/02-sauces.md": SAUCES_MD,
        "weeknight-rescue-meals/01-fridge-raid.md": FRIDGE_MD,
        "weeknight-rescue-meals/02-fifteen-minutes.md": FIFTEEN_MD,
    }
    for rel, content in files.items():
        path = directory / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return directory


def write_voice(directory: Path) -> Path:
    (directory / "samples").mkdir(parents=True, exist_ok=True)
    (directory / "style.md").write_text(STYLE_MD, encoding="utf-8")
    for name, (title, fmt, body) in SAMPLES.items():
        text = f'---\ntitle: "{title}"\nformat: {fmt}\n---\n\n{body}\n'
        (directory / "samples" / name).write_text(text, encoding="utf-8")
    return directory


class CountingEmbeddings:
    """Hashing embeddings that record every batch sent to the 'provider'."""

    def __init__(self) -> None:
        self.inner = HashingEmbeddings()
        self.calls: list[list[str]] = []

    @property
    def model(self) -> str:
        return self.inner.model

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        return self.inner.embed(texts)

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)


_PASSAGE = re.compile(
    r"^\[(\d+)\] Source: ([^\n]+)\n(.*?)(?=\n\n\[\d+\] Source: |\n\n---\n|\Z)",
    re.MULTILINE | re.DOTALL,
)


def passages_in_prompt(user: str) -> list[tuple[int, str, str]]:
    """``(number, source label, text)`` for every numbered passage in an ideas prompt."""
    return [(int(n), label, text.strip()) for n, label, text in _PASSAGE.findall(user)]


def first_long_sentence(text: str, words: int = 8) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence.split()) >= words:
            return sentence
    return text


def idea_for(number: int, text: str, fmt: str = "post") -> dict[str, Any]:
    """A well-formed idea for one passage, quoting its first long sentence."""
    return {
        "passage": number,
        "format": fmt,
        "title": f"Idea {number}",
        "hook": f"Hook {number}",
        "outline": ["one", "two", "three"],
        "quote": first_long_sentence(text),
    }


@dataclass
class FakeLLM:
    """Scripted chat model that works from the numbered passages in the prompt.

    Ideas: one per passage, quoting its first long sentence (``idea_edit`` can change the list).
    Scripts: a fixed reply (``script_reply``). ``responder`` overrides everything.
    """

    responder: Callable[[str, str], str] | None = None
    idea_edit: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None
    script_reply: dict[str, str] = field(
        default_factory=lambda: {"script": "Cook once. Eat twice.", "caption": "Sunday plan"}
    )
    fmt: str = "post"
    model_name: str = "fake-llm"
    input_tokens: int = 120
    output_tokens: int = 30
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.model_name

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.responder is not None:
            text = self.responder(system, user)
        elif schema is not None and "ideas" in schema["properties"]:
            ideas = [idea_for(n, t, self.fmt) for n, _label, t in passages_in_prompt(user)]
            if self.idea_edit is not None:
                ideas = self.idea_edit(ideas)
            text = json.dumps({"ideas": ideas})
        else:
            text = json.dumps(self.script_reply)
        return LLMResponse(
            text=text,
            model=self.model_name,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    return write_docs(tmp_path / "library")


@pytest.fixture
def voice_dir(tmp_path: Path) -> Path:
    return write_voice(tmp_path / "voice")


@pytest.fixture
def settings(tmp_path: Path, docs_dir: Path, voice_dir: Path) -> Settings:
    return Settings(
        docs_dir=docs_dir,
        voice_dir=voice_dir,
        index_path=tmp_path / "index.sqlite",
        embedding_provider="hashing",
        min_similarity=0.2,
        top_k=3,
        ideas_seed=7,
    )


@pytest.fixture
def embeddings() -> CountingEmbeddings:
    return CountingEmbeddings()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings: Settings, embeddings: CountingEmbeddings, fake_llm: FakeLLM) -> Services:
    svc = build_services(settings, embedding_provider=embeddings, llm=fake_llm)
    svc.ingestor.run()
    return svc


def with_settings(settings: Settings, **changes: object) -> Settings:
    return replace(settings, **changes)  # type: ignore[arg-type]


def chunk_ids_by_heading(services: Services) -> dict[str, str]:
    """Chunk id for each section heading of the fixture library."""
    return {c.heading: c.id for c in services.retriever.all_chunks()}


def new_idea(chunk: Chunk, fmt: str = "post", title: str = "An idea") -> NewIdea:
    return NewIdea(
        chunk_id=chunk.id,
        doc_path=chunk.doc_path,
        doc_title=chunk.doc_title,
        heading=chunk.heading,
        format=fmt,
        title=title,
        hook="A hook",
        outline=["a", "b"],
        quote=first_long_sentence(chunk.text),
    )
