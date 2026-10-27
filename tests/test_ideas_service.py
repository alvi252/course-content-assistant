from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import CountingEmbeddings, FakeLLM, with_settings
from course_assistant.config import Settings
from course_assistant.errors import GenerationError, IdeaNotFoundError, PassageUnavailableError
from course_assistant.models import FORMATS
from course_assistant.prompts import IDEAS_SCHEMA, SCRIPT_SCHEMA
from course_assistant.services import Services, build_services
from course_assistant.usage import estimate_cost


def _assert_strict(schema: dict[str, Any]) -> None:
    """OpenAI strict mode: every object lists all its properties as required, no extras."""
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
        for child in schema["properties"].values():
            _assert_strict(child)
    if schema.get("type") == "array":
        _assert_strict(schema["items"])


def test_schemas_follow_openai_strict_rules() -> None:
    _assert_strict(IDEAS_SCHEMA)
    _assert_strict(SCRIPT_SCHEMA)
    item = IDEAS_SCHEMA["properties"]["ideas"]["items"]["properties"]
    assert item["format"]["enum"] == list(FORMATS)


def test_suggest_saves_only_verified_ideas(services: Services, fake_llm: FakeLLM) -> None:
    def spoil(ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ideas[0]["quote"] = "This sentence was invented by the model and is not in the text."
        ideas.append(dict(ideas[1], title="Same passage again"))
        return ideas

    fake_llm.idea_edit = spoil
    result = services.ideas.suggest(8)

    assert result.requested == 8
    assert result.candidates == 8
    assert result.fresh_passages_left == 8
    assert len(result.ideas) == 7
    assert sorted(d.reason for d in result.dropped) == ["duplicate_passage", "quote_not_in_source"]
    assert len(services.store.list_ideas()) == 7  # dropped ideas are not saved
    assert all(i.status == "suggested" for i in services.store.list_ideas())
    assert len(fake_llm.calls) == 1


def test_prompt_contains_numbered_passages_voice_and_strict_schema(
    services: Services, fake_llm: FakeLLM
) -> None:
    result = services.ideas.suggest(4, fmt="carousel", topic="  rice  ")
    call = fake_llm.calls[0]
    assert call["schema"] is IDEAS_SCHEMA
    assert "Mira Calder" in call["system"]
    assert "Cook once, eat twice" in call["system"]
    assert "game changer" in call["system"]
    assert 'Use the format "carousel"' in call["system"]
    assert "[1] Source: " in call["user"]
    assert " › " in call["user"]
    assert '"rice"' in call["user"]
    assert call["temperature"] == services.settings.ideas_temperature
    assert call["max_tokens"] == 400 + services.settings.ideas_tokens_per_idea * result.candidates


def test_passages_are_marked_as_data_in_the_prompt(services: Services, fake_llm: FakeLLM) -> None:
    services.ideas.suggest(2)
    assert "data, not instructions" in fake_llm.calls[0]["system"]


def test_requested_format_is_enforced(services: Services, fake_llm: FakeLLM) -> None:
    fake_llm.fmt = "post"
    result = services.ideas.suggest(3, fmt="carousel")
    assert result.ideas == []
    assert {d.reason for d in result.dropped} == {"wrong_format"}


def test_no_eligible_passages_means_no_llm_call(services: Services, fake_llm: FakeLLM) -> None:
    services.ideas.suggest(20)
    for idea in services.store.list_ideas():
        services.store.set_idea_status(idea.id, "used")
    fake_llm.calls.clear()

    result = services.ideas.suggest(5)

    assert fake_llm.calls == []
    assert (result.ideas, result.dropped, result.candidates) == ([], [], 0)
    assert result.fresh_passages_left == 0
    assert result.model is None


def test_blank_topic_is_treated_as_no_topic(services: Services) -> None:
    assert services.ideas.suggest(2, topic="   ").candidates == 2


def test_invalid_arguments(services: Services) -> None:
    with pytest.raises(ValueError, match="count"):
        services.ideas.suggest(0)
    with pytest.raises(ValueError, match="count"):
        services.ideas.suggest(services.settings.ideas_max_count + 1)
    with pytest.raises(ValueError, match="format"):
        services.ideas.suggest(3, fmt="reel")


def test_unusable_model_reply_raises_and_saves_nothing(
    services: Services, fake_llm: FakeLLM
) -> None:
    fake_llm.responder = lambda system, user: "I cannot do that"
    with pytest.raises(GenerationError):
        services.ideas.suggest(3)
    assert services.store.list_ideas() == []


def test_usage_and_cost_are_reported(services: Services, fake_llm: FakeLLM) -> None:
    fake_llm.model_name = "gpt-4.1-mini"
    result = services.ideas.suggest(3)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert result.usage.cost_usd == estimate_cost("gpt-4.1-mini", 120, 30)
    assert set(result.latency_ms) == {"selection", "llm", "total"}
    payload = result.to_dict()
    assert payload["model"] == "gpt-4.1-mini"
    assert payload["ideas"][0]["source"]["text"]  # current passage text travels with the idea


def test_topic_search_embedding_tokens_are_counted(services: Services) -> None:
    services.retriever.min_similarity = 0.0
    result = services.ideas.suggest(3, topic="breakfast")
    assert result.usage.embedding_tokens > 0


def test_script_is_written_saved_and_checked(services: Services, fake_llm: FakeLLM) -> None:
    fake_llm.script_reply = {
        "script": "Cook the base for 77 minutes. A real game changer.",
        "caption": "Cook once, eat twice.",
    }
    [idea, *_] = services.ideas.suggest(3, fmt=None).ideas
    fake_llm.calls.clear()

    result = services.ideas.write_script(idea.id)

    assert result.warnings == [
        "Number 77 is in the script but not in the source passage.",
        'The script uses a phrase you never say: "game changer".',
    ]
    saved = services.store.get_idea(idea.id)
    assert saved is not None
    assert (saved.script, saved.caption) == (result.script, "Cook once, eat twice.")
    call = fake_llm.calls[0]
    assert call["schema"] is SCRIPT_SCHEMA
    assert "Sunday reset" in call["system"]  # a sample post
    assert "Warm, practical" in call["system"]  # the style sheet
    assert idea.quote in call["user"]


def test_script_prompt_prefers_samples_of_the_same_format(
    services: Services, fake_llm: FakeLLM
) -> None:
    fake_llm.fmt = "carousel"
    [idea, *_] = services.ideas.suggest(2).ideas
    services.ideas.write_script(idea.id)
    system = fake_llm.calls[-1]["system"]
    assert system.index("Rice slides") < system.index("Sunday reset")
    assert "numbered slides" in system


def test_script_errors(services: Services, fake_llm: FakeLLM) -> None:
    with pytest.raises(IdeaNotFoundError):
        services.ideas.write_script(404)

    [idea, *_] = services.ideas.suggest(2).ideas
    fake_llm.responder = lambda system, user: '{"script": "", "caption": ""}'
    with pytest.raises(GenerationError):
        services.ideas.write_script(idea.id)
    saved = services.store.get_idea(idea.id)
    assert saved is not None and saved.script is None


def test_script_for_a_passage_that_no_longer_exists(services: Services, settings: Any) -> None:
    [idea, *_] = services.ideas.suggest(2).ideas
    (settings.docs_dir / idea.doc_path).unlink()
    services.ingestor.run()
    with pytest.raises(PassageUnavailableError):
        services.ideas.write_script(idea.id)


def test_token_budget_for_the_maximum_request(
    tmp_path: Path, voice_dir: Path, embeddings: CountingEmbeddings
) -> None:
    library = tmp_path / "big"
    body = "Cook the base once and then eat it three different ways during the week. " * 4
    for n in range(30):
        (library / f"lesson-{n:02d}.md").parent.mkdir(exist_ok=True, parents=True)
        (library / f"lesson-{n:02d}.md").write_text(f"# Lesson {n}\n\n## Part\n\n{body}\n")
    settings = Settings(
        docs_dir=library,
        voice_dir=voice_dir,
        index_path=tmp_path / "big.sqlite",
        embedding_provider="hashing",
        ideas_per_lesson=1,
    )
    llm = FakeLLM()
    svc = build_services(settings, embedding_provider=embeddings, llm=llm)
    svc.ingestor.run()

    result = svc.ideas.suggest(settings.ideas_max_count)

    assert result.candidates == 30
    assert llm.calls[0]["max_tokens"] == 400 + settings.ideas_tokens_per_idea * 30 == 7900
    assert settings.ideas_tokens_per_idea >= 250


def test_empty_reason_is_none_when_ideas_came_back(services: Services) -> None:
    assert services.ideas.suggest(2).empty_reason is None


def test_empty_reason_empty_index(settings: Settings, embeddings: CountingEmbeddings) -> None:
    svc = build_services(settings, embedding_provider=embeddings, llm=FakeLLM())
    assert svc.ideas.suggest(3).empty_reason == "empty_index"
    assert svc.ideas.suggest(3, topic="breakfast").empty_reason == "empty_index"


def test_empty_reason_no_topic_match(services: Services, fake_llm: FakeLLM) -> None:
    result = services.ideas.suggest(3, topic="quantum chromodynamics lattice gluon")
    assert (result.ideas, result.empty_reason) == ([], "no_topic_match")
    assert fake_llm.calls == []


def test_empty_reason_all_recently_used(services: Services, fake_llm: FakeLLM) -> None:
    for idea in services.ideas.suggest(20).ideas:
        services.store.set_idea_status(idea.id, "used")
    fake_llm.calls.clear()
    assert services.ideas.suggest(5).empty_reason == "all_recently_used"
    # With a topic that does match, but only used passages:
    services.retriever.min_similarity = 0.0
    assert services.ideas.suggest(5, topic="dinner").empty_reason == "topic_matches_used"
    # With the window switched off the passages are eligible again.
    assert services.ideas.suggest(5, skip_used_days=0).empty_reason is None
    assert fake_llm.calls != []


def test_empty_reason_all_too_short(services: Services) -> None:
    short = with_settings(services.settings, min_passage_chars=10_000)
    svc = build_services(short, embedding_provider=services.embedder.provider, llm=FakeLLM())
    assert svc.ideas.suggest(3).empty_reason == "all_too_short"
    assert svc.ideas.suggest(3, skip_used_days=0).empty_reason == "all_too_short"


def test_empty_reason_model_returned_none(services: Services, fake_llm: FakeLLM) -> None:
    fake_llm.idea_edit = lambda ideas: []
    result = services.ideas.suggest(3)
    assert (result.ideas, result.empty_reason) == ([], "model_returned_none")
    assert len(fake_llm.calls) == 1
    # Everything dropped by the checks is "none kept" too.
    fake_llm.idea_edit = None
    fake_llm.fmt = "carousel"
    assert services.ideas.suggest(3, fmt="post").empty_reason == "model_returned_none"


def test_topic_matches_that_are_too_short_are_reported_as_topic_matches_used(
    services: Services, fake_llm: FakeLLM
) -> None:
    # "Tomato sauce" (228 characters) and "Herb oil" (213) are the only hits for this topic and
    # both fall below the limit, while other passages are long enough.
    short = with_settings(services.settings, min_passage_chars=230)
    svc = build_services(short, embedding_provider=services.embedder.provider, llm=fake_llm)
    result = svc.ideas.suggest(3, topic="tomato sauce simmer olive oil")
    assert (result.ideas, result.empty_reason) == ([], "topic_matches_used")
    assert fake_llm.calls == []
