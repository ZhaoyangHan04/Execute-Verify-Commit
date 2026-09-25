"""Native action values for the pinned STATE-Bench workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


@dataclass(frozen=True)
class StateBenchToolCall:
    """One assistant-requested STATE-Bench domain tool call.

    STATE-Bench's public ``AgentToolCallRequest`` has no transport ID.  The
    workflow assigns one before settlement so retries can be idempotent.
    """

    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        projected = project_tool_call(self)
        object.__setattr__(self, "arguments", projected["arguments"])


def project_tool_call(call: Any) -> dict[str, Any]:
    """Return the complete stable projection of a native tool call."""

    call_id = _field(call, "id")
    name = _field(call, "name")
    arguments = _field(call, "arguments")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("STATE-Bench tool call id must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("STATE-Bench tool call name must be a non-empty string")
    if not isinstance(arguments, Mapping):
        raise TypeError("STATE-Bench tool call arguments must be a mapping")
    return {
        "id": call_id,
        "name": name,
        "arguments": _json_clone(dict(arguments)),
    }


@dataclass(frozen=True)
class StateBenchToolBatch:
    """The ordered tool calls from one Producer model response."""

    calls: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.calls, tuple):
            raise TypeError("StateBenchToolBatch.calls must be a tuple")
        if not self.calls:
            raise ValueError("StateBenchToolBatch.calls must not be empty")
        call_ids = tuple(item["id"] for item in self.as_json())
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("STATE-Bench tool call ids must be unique within a batch")

    @property
    def call_ids(self) -> tuple[str, ...]:
        return tuple(item["id"] for item in self.as_json())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.as_json())

    def as_json(self) -> list[dict[str, Any]]:
        return [project_tool_call(call) for call in self.calls]

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
