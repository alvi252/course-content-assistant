from __future__ import annotations

from pathlib import Path

import pytest

from course_assistant.cli import main
from course_assistant.config import Settings


@pytest.fixture
def cli_env(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> pytest.MonkeyPatch:
    """Point the CLI at the fixture library; nothing real is read (no .env, no keys)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIBRARY_DIR", str(settings.docs_dir))
    monkeypatch.setenv("VOICE_DIR", str(settings.voice_dir))
    monkeypatch.setenv("INDEX_PATH", str(settings.index_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return monkeypatch


def test_ingest_without_an_embedding_key_prints_a_plain_message(
    cli_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_env.setenv("EMBEDDING_PROVIDER", "openai")
    assert main(["ingest"]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "No API key configured for OpenAI"
    assert "Traceback" not in captured.err + captured.out


def test_ideas_without_an_llm_key_prints_a_plain_message(
    cli_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ingest"]) == 0
    capsys.readouterr()
    assert main(["ideas", "--count", "3"]) == 1
    captured = capsys.readouterr()
    assert "No API key configured for OpenAI" in captured.err
    assert "Traceback" not in captured.err


def test_eval_without_an_llm_key_prints_a_plain_message(
    cli_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["ingest"]) == 0
    capsys.readouterr()
    assert main(["eval", "--rounds", "1", "--out", str(tmp_path / "out")]) == 1
    captured = capsys.readouterr()
    assert "No API key configured for OpenAI" in captured.err
    assert "Traceback" not in captured.err


def test_ideas_on_an_index_from_another_embedding_model(
    cli_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ingest"]) == 0
    capsys.readouterr()
    cli_env.setenv("EMBEDDING_PROVIDER", "openai")  # the index was built with hashing
    assert main(["ideas", "--topic", "breakfast"]) == 1
    captured = capsys.readouterr()
    assert "Run ingest --rebuild" in captured.err
    assert "Traceback" not in captured.err


def test_ideas_on_an_empty_index_says_so(
    cli_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ideas"]) == 2
    assert "index is empty" in capsys.readouterr().err


def test_ideas_prints_a_message_for_each_empty_reason(
    cli_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from typing import get_args

    from course_assistant.cli import EMPTY_MESSAGES
    from course_assistant.models import EmptyReason

    assert set(EMPTY_MESSAGES) == set(get_args(EmptyReason))

    assert main(["ingest"]) == 0
    capsys.readouterr()
    cli_env.setenv("MIN_PASSAGE_CHARS", "10000")  # no LLM call is made, so no key is needed
    assert main(["ideas"]) == 0
    assert EMPTY_MESSAGES["all_too_short"] in capsys.readouterr().out
    assert main(["ideas", "--topic", "quantum chromodynamics lattice gluon"]) == 0
    assert EMPTY_MESSAGES["no_topic_match"] in capsys.readouterr().out
