from __future__ import annotations

from pathlib import Path

from course_assistant.voice import (
    load_samples,
    load_style,
    never_say_hits,
    numbers_missing_from_source,
    parse_style_sheet,
    script_warnings,
)

SOURCE = "Cook the rice for 20 minutes. Keep it up to 4 days in the fridge."


def test_style_sheet_lists_are_parsed(voice_dir: Path) -> None:
    style = load_style(voice_dir)
    assert style.phrases == [
        "Cook once, eat twice",
        "Here is the thing",
        "Keep it boring, keep it easy",
    ]
    assert style.never_say == ["game changer", "guilt-free", "clean eating"]
    assert "Warm, practical" in style.raw


def test_missing_style_sheet_gives_an_empty_one(tmp_path: Path) -> None:
    style = load_style(tmp_path)
    assert (style.raw, style.phrases, style.never_say) == ("", [], [])


def test_list_headings_are_matched_loosely() -> None:
    sheet = parse_style_sheet("### Phrases I use (often)\n* one\n\n# NEVER SAY\n1. two\n")
    assert (sheet.phrases, sheet.never_say) == (["one"], ["two"])


def test_samples_are_loaded_with_title_and_format(voice_dir: Path) -> None:
    samples = load_samples(voice_dir)
    assert [(s.path, s.title, s.format) for s in samples] == [
        ("01-sunday-reset.md", "Sunday reset", "post"),
        ("02-rice-slides.md", "Rice slides", "carousel"),
        ("03-fridge-talk.md", "Fridge talk", "talking_head"),
    ]
    assert samples[0].text.startswith("Here is the thing")


def test_no_samples_folder(tmp_path: Path) -> None:
    assert load_samples(tmp_path) == []


def test_numbers_absent_from_the_source_are_found() -> None:
    assert numbers_missing_from_source("Cook it for 90 minutes, keep 4 days.", SOURCE) == ["90"]
    assert numbers_missing_from_source("20 minutes", SOURCE) == []


def test_list_numbering_is_not_a_claim() -> None:
    script = "1. Cook the rice\n2. Cool it\nSlide 3: Eat it\nStep 4 is easy"
    assert numbers_missing_from_source(script, "no digits") == []


def test_never_say_is_case_insensitive() -> None:
    assert never_say_hits("A GAME CHANGER, truly", ["game changer", "hack"]) == ["game changer"]


def test_script_warnings_combine_both_checks(voice_dir: Path) -> None:
    style = load_style(voice_dir)
    warnings = script_warnings(
        "Ready in 90 minutes, a game changer.", "Keeps 4 days.", SOURCE, style.never_say
    )
    assert warnings == [
        "Number 90 is in the script but not in the source passage.",
        'The script uses a phrase you never say: "game changer".',
    ]
    assert script_warnings("Clean script, 20 minutes.", "Fine.", SOURCE, style.never_say) == []
