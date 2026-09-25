"""Native τ² action values used by the workflow adapter.

The Shadow kernel deliberately knows nothing about tool calls.  This module
keeps the complete assistant tool-call batch as the workflow's atomic action
without translating it into framework-specific transaction types.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def project_tool_call(call: Any) -> dict[str, Any]:
    """Return the stable, allowlisted projection of one native tool call."""

    call_id = _field(call, "id")
    name = _field(call, "name")
    arguments = _field(call, "arguments")
    requestor = _field(call, "requestor") or "assistant"
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tau2 tool call id must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("tau2 tool call name must be a non-empty string")
    if not isinstance(arguments, Mapping):
        raise TypeError("tau2 tool call arguments must be a mapping")
    if requestor != "assistant":
        raise ValueError("Tau2ToolBatch accepts assistant tool calls only")
    projected = {
        "id": call_id,
        "name": name,
        "arguments": dict(arguments),
        "requestor": requestor,
    }
    # Fail early if a provider produced arguments outside the native JSON
    # tool-call envelope.  Repr-based fingerprints would not be stable.
    json.dumps(projected, ensure_ascii=False, sort_keys=True)
    return projected


@dataclass(frozen=True)
class Tau2ToolBatch:
    """One complete assistant message's ordered native tool calls."""

    calls: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.calls, tuple):
            raise TypeError("Tau2ToolBatch.calls must be a tuple")
        if not self.calls:
            raise ValueError("Tau2ToolBatch.calls must not be empty")
        projected = tuple(project_tool_call(call) for call in self.calls)
        call_ids = tuple(item["id"] for item in projected)
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("tau2 tool call ids must be unique within a batch")

    @property
    def call_ids(self) -> tuple[str, ...]:
        return tuple(project_tool_call(call)["id"] for call in self.calls)

    def as_json(self) -> list[dict[str, Any]]:
        """Return an ordered JSON-compatible representation."""

        return [project_tool_call(call) for call in self.calls]

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
