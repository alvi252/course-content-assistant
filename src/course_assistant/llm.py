"""Chat-completion providers behind one small interface."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import ProviderError, ProviderNotConfiguredError

JsonSchema = dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: JsonSchema | None = None,
    ) -> LLMResponse:
        """Return the model's text. With ``schema``, the text is JSON matching it."""
        ...


def make_client(provider: str, factory: Callable[[], Any]) -> Any:
    """Create an SDK client; a missing key or SDK becomes a clear, key-free ProviderError."""
    try:
        return factory()
    except ImportError as exc:
        raise ProviderNotConfiguredError(provider, f"The {provider} SDK is not installed") from exc
    except Exception as exc:  # the SDKs raise their own error type when the key is missing
        raise ProviderNotConfiguredError(provider) from exc


def _anthropic_client(timeout_s: float) -> Any:
    from anthropic import Anthropic

    return Anthropic(timeout=timeout_s, max_retries=2)  # reads ANTHROPIC_API_KEY


class OpenAIChat:
    """OpenAI Chat Completions with Structured Outputs (tested live with ``gpt-4.1-mini``)."""

    def __init__(
        self, model: str = "gpt-4.1-mini", *, timeout_s: float = 30.0, client: Any | None = None
    ) -> None:
        self._model = model
        if client is None:
            from openai import OpenAI

            client = make_client(
                "OpenAI", lambda: OpenAI(timeout=timeout_s, max_retries=2)
            )  # reads OPENAI_API_KEY
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: JsonSchema | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "strict": True, "schema": schema},
            }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError("OpenAI", exc) from exc
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model or self._model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


class AnthropicChat:
    """Anthropic Messages API with structured JSON output.

    Implemented and unit-tested against a stub client, but not exercised live in this demo.
    Install the extra (``pip install '.[anthropic]'``) and set ``ANTHROPIC_API_KEY``.
    ``temperature`` is accepted for interface compatibility but not sent: the current Messages
    API exposes no sampling-temperature parameter.
    """

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        *,
        timeout_s: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        if client is None:
            client = make_client("Anthropic", lambda: _anthropic_client(timeout_s))
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: JsonSchema | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ProviderError("Anthropic", exc) from exc
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = response.usage
        return LLMResponse(
            text=text,
            model=getattr(response, "model", None) or self._model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


def build_llm_provider(name: str, model: str, timeout_s: float) -> LLMProvider:
    if name == "openai":
        return OpenAIChat(model, timeout_s=timeout_s)
    if name == "anthropic":
        return AnthropicChat(model, timeout_s=timeout_s)
    raise ValueError(f"Unknown LLM_PROVIDER: {name!r} (expected 'openai' or 'anthropic')")


class LazyLLM:
    """Builds the real provider on the first call, so the app starts without an API key.

    Listing, coverage and the status buttons never need the model; only generation does.
    """

    def __init__(self, name: str, model: str, timeout_s: float) -> None:
        if name not in {"openai", "anthropic"}:
            raise ValueError(f"Unknown LLM_PROVIDER: {name!r} (expected 'openai' or 'anthropic')")
        self._name = name
        self._model = model
        self._timeout_s = timeout_s
        self._provider: LLMProvider | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: JsonSchema | None = None,
    ) -> LLMResponse:
        with self._lock:
            if self._provider is None:
                self._provider = build_llm_provider(self._name, self._model, self._timeout_s)
            provider = self._provider
        return provider.complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )
