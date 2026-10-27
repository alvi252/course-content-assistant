from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CountingEmbeddings, FakeLLM, new_idea, with_settings
from course_assistant.config import Settings
from course_assistant.eval import render_markdown, run_on_copy, write_reports
from course_assistant.services import build_services


def test_eval_runs_on_a_copy_and_writes_both_reports(
    settings: Settings, embeddings: CountingEmbeddings, tmp_path: Path
) -> None:
    settings = with_settings(settings, ideas_per_lesson=1, min_similarity=0.0)
    real = build_services(settings, embedding_provider=embeddings, llm=FakeLLM())
    real.ingestor.run()
    chunk = real.retriever.all_chunks()[0]
    [kept] = real.store.add_ideas([new_idea(chunk)])
    llm = FakeLLM(model_name="gpt-4.1-mini")

    report = run_on_copy(settings, rounds=3, embedding_provider=embeddings, llm=llm)

    # The real ideas log is exactly as it was.
    assert [i.id for i in real.store.list_ideas()] == [kept.id]
    assert real.store.list_ideas()[0].status == "suggested"

    # Four lessons, one passage each per round: rounds 1 and 2 are full, round 3 has nothing.
    r1, r2, r3 = report.rounds
    assert (r1.proposed, r1.kept, r1.grounded_rate) == (4, 4, 1.0)
    assert (r1.distinct_lessons, r1.from_new_lessons, r1.repeats) == (4, 4, 0)
    assert (r2.kept, r2.from_new_lessons, r2.repeats) == (4, 0, 0)
    assert (r3.proposed, r3.kept, r3.grounded_rate) == (0, 0, 0.0)
    assert r1.fresh_passages_left == 8 and r3.fresh_passages_left == 0

    assert report.topic is not None
    assert (report.topic.requested, report.topic.kept) == (5, 4)
    assert 0 <= report.topic.mention_topic <= report.topic.kept
    assert len(report.scripts) == 3
    assert report.totals["llm_calls"] == 2 + 1 + 3  # round 3 had nothing to ask about
    assert report.totals["input_tokens"] == 6 * 120
    assert report.totals["cost_usd"] is None  # hashing embeddings have no known price

    md_path, json_path = write_reports(report, tmp_path / "eval")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert [r["kept"] for r in data["rounds"]] == [4, 4, 0]
    assert data["topic"]["topic"] == "leftovers"
    assert len(data["scripts"]) == 3
    text = md_path.read_text(encoding="utf-8")
    assert text == render_markdown(report)
    assert "| 1 | 4 | 4 | 100% |" in text
    assert "Repeats must be 0" in text
    assert "Estimated cost: n/a" in text


def test_eval_reports_dropped_reasons(settings: Settings, embeddings: CountingEmbeddings) -> None:
    settings = with_settings(settings, min_similarity=0.0)
    build_services(settings, embedding_provider=embeddings, llm=FakeLLM()).ingestor.run()

    def spoil(ideas: list[dict[str, object]]) -> list[dict[str, object]]:
        ideas[0]["quote"] = "A sentence that the model made up and that is nowhere in the text."
        return ideas

    report = run_on_copy(
        settings, rounds=1, embedding_provider=embeddings, llm=FakeLLM(idea_edit=spoil)
    )
    first = report.rounds[0]
    assert first.dropped == {"quote_not_in_source": 1}
    assert first.grounded_rate == round(first.kept / first.proposed, 3) < 1.0


def test_eval_needs_an_index(settings: Settings, embeddings: CountingEmbeddings) -> None:
    with pytest.raises(FileNotFoundError):
        run_on_copy(settings, embedding_provider=embeddings, llm=FakeLLM())


def test_mention_topic_counts_only_passages_with_the_word(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    settings = with_settings(settings, ideas_per_lesson=1, min_similarity=0.0)
    build_services(settings, embedding_provider=embeddings, llm=FakeLLM()).ingestor.run()
    report = run_on_copy(settings, rounds=1, embedding_provider=embeddings, llm=FakeLLM())
    assert report.topic is not None
    # One lesson per idea, four lessons: four kept. The passages that say "leftover" are in two
    # lessons ("Herb oil" and "Rescue the leftovers"); the other two never use the word.
    assert report.topic.kept == 4
    assert report.topic.mention_topic == 2
    expected = f"2 of the kept ideas come from passages that mention '{report.topic.topic}'"
    assert expected in render_markdown(report)
