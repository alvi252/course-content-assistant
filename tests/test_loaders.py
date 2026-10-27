from __future__ import annotations

from pathlib import Path

import pytest

from course_assistant.loaders import (
    UnsupportedFileError,
    iter_source_files,
    load_document,
    parse_markdown,
)


def test_markdown_title_and_heading_paths() -> None:
    raw = (
        "# Care guide\n\nIntro text.\n\n## Fountain\n\n### Filter\n\nChange it.\n\n"
        "## Feeder\n\nClean it."
    )
    title, sections = parse_markdown(raw, "fallback")
    assert title == "Care guide"
    assert [s.heading for s in sections] == ["Overview", "Fountain › Filter", "Feeder"]
    assert sections[1].text == "Change it."


def test_front_matter_title_wins_over_h1() -> None:
    raw = "---\ntitle: Official Title\nauthor: x\n---\n# Page heading\n\nBody."
    title, sections = parse_markdown(raw, "fallback")
    assert title == "Official Title"
    assert sections[0].heading == "Overview"


def test_inline_markup_and_comments_are_cleaned() -> None:
    raw = (
        "# T\n\nSee [our policy](https://example.com/p) and ![a dog](dog.png).\n"
        "<!-- internal note -->\nThis is **important**."
    )
    _, sections = parse_markdown(raw, "fallback")
    text = sections[0].text
    assert "our policy" in text and "https://" not in text
    assert "a dog" in text and "dog.png" not in text
    assert "internal note" not in text
    assert "important" in text and "**" not in text


def test_headings_inside_code_fences_are_not_sections() -> None:
    raw = "# T\n\n## Setup\n\n```bash\n# not a heading\necho hi\n```\n"
    _, sections = parse_markdown(raw, "fallback")
    assert [s.heading for s in sections] == ["Setup"]
    assert "# not a heading" in sections[0].text


def test_text_without_headings_falls_back_to_filename(tmp_path: Path) -> None:
    (tmp_path / "opening-hours.txt").write_text("We open at 9.\n\n\n\nWe close at 18.")
    doc = load_document(tmp_path / "opening-hours.txt", tmp_path)
    assert doc.title == "Opening hours"
    assert doc.sections[0].text == "We open at 9.\n\nWe close at 18."
    assert len(doc.content_hash) == 64


def _minimal_pdf(text: str) -> bytes:
    """Build a one-page PDF with a single line of Helvetica text."""
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def test_pdf_pages_become_sections(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    (tmp_path / "warranty.pdf").write_bytes(_minimal_pdf("Warranty lasts two years"))
    doc = load_document(tmp_path / "warranty.pdf", tmp_path)
    assert doc.title == "Warranty"
    assert doc.sections[0].heading == "Page 1"
    assert "Warranty lasts two years" in doc.sections[0].text


def test_broken_pdf_raises_unsupported(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf at all")
    with pytest.raises(UnsupportedFileError):
        load_document(tmp_path / "broken.pdf", tmp_path)


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    (tmp_path / "sheet.xlsx").write_bytes(b"x")
    with pytest.raises(UnsupportedFileError):
        load_document(tmp_path / "sheet.xlsx", tmp_path)


def test_iter_source_files_skips_hidden_and_unsupported(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.txt").write_text("B")
    (tmp_path / "c.png").write_bytes(b"")
    (tmp_path / ".hidden.md").write_text("# H")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.md").write_text("# X")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.md").write_text("# D")
    names = [p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path)]
    assert names == ["a.md", "b.txt", "sub/d.md"]
