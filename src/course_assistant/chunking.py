"""Heading-aware chunking with a sentence-aligned overlap between neighbouring chunks."""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, LoadedDocument

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split on word boundaries; fall back to characters for absurdly long tokens."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Break an oversized paragraph into sentence groups that each fit ``max_chars``."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    groups: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        parts = [sentence] if len(sentence) <= max_chars else _hard_split(sentence, max_chars)
        for part in parts:
            candidate = f"{current} {part}" if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                groups.append(current)
                current = part
    if current:
        groups.append(current)
    return groups


def overlap_tail(text: str, overlap_chars: int) -> str:
    """Return the longest suffix of ``text`` that starts at a sentence (or word) boundary
    and is at most ``overlap_chars`` long.

    >>> overlap_tail("One. Two three. Four five six.", 20)
    'Four five six.'
    >>> overlap_tail("no sentence boundaries here at all", 10)
    'at all'
    >>> overlap_tail("anything", 0)
    ''
    """
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text
    window = text[-overlap_chars:]
    starts = [m.end() for m in _SENTENCE_END.finditer(text)]
    cut_min = len(text) - overlap_chars
    sentence_starts = [s for s in starts if s >= cut_min]
    if sentence_starts:
        return text[sentence_starts[0] :].strip()
    space = window.find(" ")
    return window[space + 1 :].strip() if space != -1 else ""


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack paragraphs into windows of at most ``max_chars`` characters.

    Consecutive windows share a sentence-aligned tail of up to ``overlap_chars`` so a fact
    that straddles a boundary is still retrievable from either side.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text.strip()):
        paragraph = paragraph.strip()
        if paragraph:
            units.extend(_split_paragraph(paragraph, max_chars))

    windows: list[str] = []
    current: list[str] = []
    for unit in units:
        if len("\n\n".join([*current, unit])) <= max_chars:
            current.append(unit)
            continue
        # `current` is never empty here: every unit fits on its own.
        windows.append("\n\n".join(current))
        tail = overlap_tail(windows[-1], overlap_chars)
        current = [tail, unit] if tail and len(tail) + 2 + len(unit) <= max_chars else [unit]
    if current:
        windows.append("\n\n".join(current))
    return windows


def chunk_id(doc_path: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(f"{doc_path}\x00{ordinal}\x00{text}".encode()).hexdigest()
    return digest[:16]


def chunk_document(doc: LoadedDocument, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Turn a loaded document into retrieval chunks. Chunks never cross a heading."""
    chunks: list[Chunk] = []
    for section in doc.sections:
        for piece in split_text(section.text, max_chars, overlap_chars):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=chunk_id(doc.path, ordinal, piece),
                    doc_path=doc.path,
                    doc_title=doc.title,
                    heading=section.heading,
                    ordinal=ordinal,
                    text=piece,
                )
            )
    return chunks
