"""Exceptions shared by the provider modules."""

from __future__ import annotations


class ProviderError(RuntimeError):
    """An upstream AI provider failed (network, auth, rate limit, bad response)."""

    def __init__(self, provider: str, cause: Exception) -> None:
        status = getattr(cause, "status_code", None)
        detail = type(cause).__name__ + (f" (HTTP {status})" if status else "")
        # The provider's own message is deliberately not copied: it can echo request details.
        super().__init__(f"{provider} request failed: {detail}")
        self.provider = provider
        self.status_code = status


class GenerationError(RuntimeError):
    """The model answered, but the reply cannot be used (not JSON, wrong shape, empty)."""


class IdeaNotFoundError(LookupError):
    """No idea with this id in the log."""


class PassageUnavailableError(RuntimeError):
    """The passage behind an idea is no longer in the index (the lesson was edited or removed)."""


class ProviderNotConfiguredError(ProviderError):
    """No usable credentials (or SDK) for a provider. The message never contains a key."""

    def __init__(self, provider: str, message: str | None = None) -> None:
        RuntimeError.__init__(self, message or f"No API key configured for {provider}")
        self.provider = provider
        self.status_code = None


class IndexModelMismatchError(RuntimeError):
    """The index was built with another embedding model than the one configured now."""
