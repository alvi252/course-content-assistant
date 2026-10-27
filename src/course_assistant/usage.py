"""Cost estimates and an optional JSONL usage log for every paid API call."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

# USD per 1M tokens (input, output). List prices at the time of writing; check your
# provider's pricing page and edit this table if they change. Unknown models -> cost None.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    """Estimate the USD cost of one call. Dated model ids match their base name.

    >>> estimate_cost("gpt-4.1-mini", 1_000_000, 1_000_000)
    2.0
    >>> estimate_cost("gpt-4o-mini-2024-07-18", 2_000, 0)
    0.0003
    >>> estimate_cost("hashing-512", 10) is None
    True
    """
    matches = [
        name for name in PRICES_USD_PER_MTOK if model == name or model.startswith(name + "-")
    ]
    if not matches:
        return None
    price_in, price_out = PRICES_USD_PER_MTOK[max(matches, key=len)]
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 8)


class UsageLog:
    """Appends one JSON line per paid call. Disabled when ``path`` is None.

    Only counts and model names are written: never prompts, answers or credentials.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(self, kind: str, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        if self.path is None:
            return
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": estimate_cost(model, input_tokens, output_tokens),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
