"""Load Markdown, plain-text and PDF files into cleaned, heading-aware sections."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from .models import LoadedDocument, Section

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
HEADING_SEPARATOR = " › "
INTRO_HEADING = "Overview"

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")
_FENCE = re.compile(r"^\s*(```|~~~)")


class UnsupportedFileError(ValueError):
    pass


def iter_source_files(docs_dir: Path) -> Iterator[Path]:
    """Yield supported files under ``docs_dir`` in a stable order, skipping hidden files."""
    for path in sorted(docs_dir.rglob("*")):
        rel_parts = path.relative_to(docs_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def title_from_filename(path: Path) -> str:
    words = re.sub(r"[-_]+", " ", path.stem).strip()
    return words[:1].upper() + words[1:] if words else path.name


def clean_text(text: str) -> str:
    """Normalise unicode and whitespace without destroying paragraph structure."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = text.replace(" ", " ").replace("​", "")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_markdown_inline(text: str) -> str:
    """Strip markup that adds noise for embeddings and snippets, keep the words."""
    text = _IMAGE.sub(lambda m: m.group(1), text)
    text = _LINK.sub(lambda m: m.group(1), text)
    text = _EMPHASIS.sub(lambda m: m.group(2), text)
    return text


def _front_matter_title(front_matter: str) -> str | None:
    for line in front_matter.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "title" and value.strip():
            return value.strip().strip("\"'")
    return None


def parse_markdown(raw: str, fallback_title: str) -> tuple[str, list[Section]]:
    """Split Markdown into sections keyed by their heading path (H1 is the document title)."""
    text = clean_text(raw)
    title: str | None = None

    match = _FRONT_MATTER.match(text + "\n")
    if match:
        title = _front_matter_title(match.group(1))
        text = text[match.end() :]
    text = _HTML_COMMENT.sub("", text)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading) below the title
    buffer: list[str] = []
    in_fence = False
    title_heading_seen = False

    def flush() -> None:
        body = clean_text("\n".join(buffer))
        buffer.clear()
        if body:
            heading = HEADING_SEPARATOR.join(h for _, h in stack) or INTRO_HEADING
            sections.append(Section(heading=heading, text=body))

    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        heading_match = None if in_fence else _HEADING.match(line)
        if heading_match is None:
            buffer.append(line if in_fence else clean_markdown_inline(line))
            continue

        flush()
        level = len(heading_match.group(1))
        heading = clean_markdown_inline(heading_match.group(2)).strip()
        if level == 1 and not title_heading_seen and not stack:
            # The first H1 is the document title (front matter, if any, takes precedence).
            title_heading_seen = True
            title = title or heading
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
    flush()

    return title or fallback_title, sections


def parse_plain_text(raw: str) -> list[Section]:
    body = clean_text(raw)
    return [Section(heading=INTRO_HEADING, text=body)] if body else []


def _load_pdf(path: Path) -> tuple[str | None, list[Section]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise UnsupportedFileError(
            f"{path.name}: PDF support needs the optional dependency: pip install '.[pdf]'"
        ) from exc

    try:
        reader = PdfReader(str(path))
        meta_title = None
        if reader.metadata is not None and reader.metadata.title:
            meta_title = str(reader.metadata.title).strip() or None
        sections = []
        for number, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            if page_text:
                sections.append(Section(heading=f"Page {number}", text=page_text))
    except Exception as exc:  # pypdf raises many types for malformed/encrypted files
        raise UnsupportedFileError(f"{path.name}: cannot read PDF ({type(exc).__name__})") from exc
    return meta_title, sections


def load_document(path: Path, docs_dir: Path) -> LoadedDocument:
    """Read one file and return its title, sections and a hash of the raw bytes."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(f"Unsupported file type: {path.name}")

    raw_bytes = path.read_bytes()
    rel_path = path.relative_to(docs_dir).as_posix()
    fallback_title = title_from_filename(path)

    if suffix == ".pdf":
        meta_title, sections = _load_pdf(path)
        title = meta_title or fallback_title
    else:
        raw = raw_bytes.decode("utf-8", errors="replace")
        if suffix == ".txt":
            title, sections = fallback_title, parse_plain_text(raw)
        else:
            title, sections = parse_markdown(raw, fallback_title)

    return LoadedDocument(
        path=rel_path,
        title=title,
        sections=sections,
        content_hash=content_hash(raw_bytes),
    )
