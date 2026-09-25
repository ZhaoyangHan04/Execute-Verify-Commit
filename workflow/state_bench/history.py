"""Separate Producer-private recovery context from canonical public history."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _validate_message(message: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise TypeError("history message must be a mapping")
    role = message.get("role")
    if role not in {"user", "assistant", "tool"}:
        raise ValueError(f"unsupported history role: {role!r}")
    return _json_clone(dict(message))


@dataclass
class StateBenchHistories:
    """Two append-only histories with an explicit rejection-only operation."""

    _producer: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _public: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @property
    def producer(self) -> tuple[dict[str, Any], ...]:
        return tuple(_json_clone(message) for message in self._producer)

    @property
    def public(self) -> tuple[dict[str, Any], ...]:
        return tuple(_json_clone(message) for message in self._public)

    def append_shared(self, *messages: Mapping[str, Any]) -> None:
        """Append the same user-visible event to both histories."""

        if not messages:
            raise ValueError("append_shared requires at least one message")
        cloned = [_validate_message(message) for message in messages]
        self._public.extend(_json_clone(message) for message in cloned)
        self._producer.extend(cloned)

    def append_producer_private(self, *messages: Mapping[str, Any]) -> None:
        """Append internal tool rounds or rejected rounds to Producer only."""

        if not messages:
            raise ValueError("append_producer_private requires at least one message")
        self._producer.extend(_validate_message(message) for message in messages)

    def append_public_only(self, *messages: Mapping[str, Any]) -> None:
        """Append the canonical compressed form of an internal Producer turn."""

        if not messages:
            raise ValueError("append_public_only requires at least one message")
        self._public.extend(_validate_message(message) for message in messages)

    def append_rejected(self, *messages: Mapping[str, Any]) -> None:
        """Append a rejected proposal and its feedback to Producer only."""

        if not messages:
            raise ValueError("append_rejected requires at least one message")
        cloned = [_validate_message(message) for message in messages]
        if cloned[0]["role"] != "assistant" or not cloned[0].get("tool_calls"):
            raise ValueError("a rejected proposal must be an assistant tool-call message")
        if any(message["role"] != "tool" for message in cloned[1:]):
            raise ValueError("rejected proposal feedback must use tool-role messages")
        self._producer.extend(cloned)
