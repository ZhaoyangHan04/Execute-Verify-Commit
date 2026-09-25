"""Provider-neutral completion values shared by optional backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCall:
    """One validated native function call returned by a model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class CompletionProvenance:
    """Non-secret facts needed to audit response interpretation."""

    requested_model: str
    response_model: str
    request_mode: Literal["text", "json", "tools"]
    mixed_content_and_tool_calls: bool
    tool_schema_count: int
    tool_call_count: int
    seed: int | None
    temperature: float
    top_p: float
    max_completion_tokens: int
    enable_thinking: bool


@dataclass(frozen=True)
class Completion:
    """Provider-neutral subset of a native chat completion."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    model: str
    usage: Usage
    latency_s: float
    raw_content: str | None
    provenance: CompletionProvenance
