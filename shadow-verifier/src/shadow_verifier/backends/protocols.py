"""Provider-neutral protocols for optional completion backends."""

from __future__ import annotations

from typing import Any, Protocol

from .model import Completion


class CompletionBackend(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> Completion: ...
