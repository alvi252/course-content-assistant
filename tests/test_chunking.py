from __future__ import annotations

from itertools import pairwise

import pytest

from course_assistant.chunking import chunk_document, overlap_tail, split_text
from course_assistant.models import LoadedDocument, Section


def _doc(*sections: Section) -> LoadedDocument:
    return LoadedDocument(path="guide.md", title="Guide", sections=list(sections), content_hash="h")


def _sentences(prefix: str, count: int) -> str:
    return " ".join(f"{prefix} sentence number {i} has a few words." for i in range(count))


def test_short_section_is_one_chunk_with_its_heading() -> None:
    chunks = chunk_document(_doc(Section("Returns", "Return within 30 days.")), 500, 100)
    assert len(chunks) == 1
    assert chunks[0].heading == "Returns"
    assert chunks[0].embed_text.startswith("Guide — Returns")


def test_chunks_never_cross_headings() -> None:
    doc = _doc(Section("A", "Alpha text."), Section("B", "Beta text."))
    chunks = chunk_document(doc, 500, 100)
    assert [(c.heading, c.text) for c in chunks] == [("A", "Alpha text."), ("B", "Beta text.")]
    assert [c.ordinal for c in chunks] == [0, 1]


def test_long_section_is_split_within_limit_with_overlap() -> None:
    paragraphs = "\n\n".join(_sentences(f"P{i}", 3) for i in range(8))
    windows = split_text(paragraphs, max_chars=400, overlap_chars=120)
    assert len(windows) > 2
    assert all(len(w) <= 400 for w in windows)
    for previous, current in pairwise(windows):
        tail = overlap_tail(previous, 120)
        assert tail, "expected a sentence-aligned overlap"
        assert current.startswith(tail)


def test_oversized_paragraph_is_split_on_sentences() -> None:
    paragraph = _sentences("Long", 30)  # one paragraph, ~1300 chars
    windows = split_text(paragraph, max_chars=300, overlap_chars=0)
    assert len(windows) > 3
    assert all(len(w) <= 300 for w in windows)
    assert all(w.endswith(".") for w in windows)  # split at sentence ends
    assert " ".join(windows) == paragraph  # nothing lost when there is no overlap


def test_giant_token_is_hard_split() -> None:
    windows = split_text("x" * 950, max_chars=400, overlap_chars=50)
    assert [len(w) for w in windows] == [400, 400, 150]


def test_chunk_ids_are_deterministic_and_unique() -> None:
    doc = _doc(Section("A", "Same text."), Section("B", "Same text."))
    first = chunk_document(doc, 500, 100)
    second = chunk_document(doc, 500, 100)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == 2


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, 150)])
def test_invalid_sizes_are_rejected(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("text", max_chars, overlap)
