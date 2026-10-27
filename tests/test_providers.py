"""Provider adapters, exercised with stub SDK clients (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from course_assistant.embeddings import HashingEmbeddings, OpenAIEmbeddings
from course_assistant.errors import ProviderError, ProviderNotConfiguredError
from course_assistant.llm import AnthropicChat, LazyLLM, OpenAIChat
from course_assistant.prompts import SCRIPT_SCHEMA
from course_assistant.usage import estimate_cost


class _Recorder:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        if self.error:
            raise self.error
        return self.response(kwargs) if callable(self.response) else self.response


def test_openai_chat_request_and_response_mapping() -> None:
    completions = _Recorder(
        SimpleNamespace(
            model="gpt-4.1-mini-2025-04-14",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"script": "x"}'))],
            usage=SimpleNamespace(prompt_tokens=900, completion_tokens=60),
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    chat = OpenAIChat("gpt-4.1-mini", client=client)

    reply = chat.complete(
        system="S", user="U", max_tokens=300, temperature=0.0, schema=SCRIPT_SCHEMA
    )

    sent = completions.kwargs[0]
    assert sent["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "strict": True, "schema": SCRIPT_SCHEMA},
    }
    assert sent["max_completion_tokens"] == 300
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ('{"script": "x"}', 900, 60)
    assert reply.model == "gpt-4.1-mini-2025-04-14"


class _FakeAuthError(Exception):
    status_code = 401


def test_provider_errors_are_wrapped_without_echoing_details() -> None:
    secret_ish = "Incorrect API key provided: fake-credential-should-not-leak"
    completions = _Recorder(error=_FakeAuthError(secret_ish))
    chat = OpenAIChat(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    with pytest.raises(ProviderError) as info:
        chat.complete(system="S", user="U", max_tokens=10, temperature=0)
    assert "fake-credential" not in str(info.value)
    assert "HTTP 401" in str(info.value)
    assert info.value.status_code == 401


def test_anthropic_chat_joins_text_blocks() -> None:
    messages = _Recorder(
        SimpleNamespace(
            model="claude-test",
            content=[
                SimpleNamespace(type="text", text='{"script": "ok", '),
                SimpleNamespace(type="thinking", text="ignored"),
                SimpleNamespace(type="text", text='"caption": "c"}'),
            ],
            usage=SimpleNamespace(input_tokens=500, output_tokens=20),
        )
    )
    chat = AnthropicChat("claude-test", client=SimpleNamespace(messages=messages))

    reply = chat.complete(
        system="S", user="U", max_tokens=200, temperature=0.0, schema=SCRIPT_SCHEMA
    )

    sent = messages.kwargs[0]
    assert sent["system"] == "S"
    assert sent["messages"] == [{"role": "user", "content": "U"}]
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": SCRIPT_SCHEMA}}
    assert "temperature" not in sent
    assert reply.text == '{"script": "ok", "caption": "c"}'
    assert (reply.input_tokens, reply.output_tokens) == (500, 20)


def test_openai_embeddings_batch_and_restore_order() -> None:
    def respond(kwargs: dict[str, Any]) -> Any:
        batch = kwargs["input"]
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0])
            for i, text in enumerate(batch)
        ]
        return SimpleNamespace(
            data=list(reversed(data)), usage=SimpleNamespace(prompt_tokens=len(batch))
        )

    embeddings_api = _Recorder(respond)
    provider = OpenAIEmbeddings(
        "text-embedding-3-small", batch_size=2, client=SimpleNamespace(embeddings=embeddings_api)
    )
    result = provider.embed(["a", "bbb", "cc"])

    assert [len(k["input"]) for k in embeddings_api.kwargs] == [2, 1]
    assert result.tokens == 3
    assert result.vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
    expected_first = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(result.vectors[0], expected_first)  # "a" -> [1, 1], order restored


def test_hashing_embeddings_are_deterministic_and_lexical() -> None:
    provider = HashingEmbeddings(dim=256)
    a, b, c = provider.embed(["fountain filter", "fountain filter", "parcel delivery"]).vectors
    assert np.allclose(a, b)
    assert float(a @ c) < 0.5 < float(a @ b)


@pytest.mark.parametrize(
    ("model", "tokens_in", "tokens_out", "expected"),
    [
        ("gpt-4.1-mini", 1000, 200, 0.00072),
        ("text-embedding-3-small", 50_000, 0, 0.001),
        ("unknown-model", 1, 1, None),
    ],
)
def test_cost_estimates(
    model: str, tokens_in: int, tokens_out: int, expected: float | None
) -> None:
    cost = estimate_cost(model, tokens_in, tokens_out)
    assert cost == (pytest.approx(expected) if expected is not None else None)


def test_lazy_llm_builds_the_provider_only_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, str]] = []

    class _Stub:
        model = "stub"

        def complete(self, **kwargs: Any) -> str:
            return "reply"

    def fake_build(name: str, model: str, timeout_s: float) -> _Stub:
        built.append((name, model))
        return _Stub()

    monkeypatch.setattr("course_assistant.llm.build_llm_provider", fake_build)
    lazy = LazyLLM("openai", "gpt-4.1-mini", 5.0)
    assert lazy.model == "gpt-4.1-mini"
    assert built == []  # no key is needed until a call is made
    lazy.complete(system="S", user="U", max_tokens=1, temperature=0)  # type: ignore[arg-type]
    lazy.complete(system="S", user="U", max_tokens=1, temperature=0)  # type: ignore[arg-type]
    assert built == [("openai", "gpt-4.1-mini")]


def test_lazy_llm_rejects_unknown_provider_at_once() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        LazyLLM("nope", "m", 5.0)


def test_missing_key_gives_a_clear_error_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfiguredError) as info:
        OpenAIChat("gpt-4.1-mini")
    assert str(info.value) == "No API key configured for OpenAI"
    assert isinstance(info.value, ProviderError)
    with pytest.raises(ProviderNotConfiguredError, match="No API key configured for OpenAI"):
        OpenAIEmbeddings().embed(["text"])


def test_openai_embeddings_do_not_need_a_key_until_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIEmbeddings().model == "text-embedding-3-small"
