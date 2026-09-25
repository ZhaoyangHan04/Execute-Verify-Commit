"""Leakage-safe evidence projection for the STATE-Bench workflow.

There is deliberately no argument for a ``TaskDefinition``, simulator config,
task summary, requirement, raw task environment, reference trajectory, or
official score.  Those values belong to the trusted final-evaluation boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from shadow_verifier import Evidence, Part
from shadow_verifier.projections import structural_diff

from .model import StateBenchToolBatch, project_tool_call


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
class StateBenchEvidenceContext:
    """The full allowlist visible to one fresh semantic reviewer."""

    agent_system_prompt: str
    public_history: tuple[Any, ...]
    tool_schemas: tuple[Mapping[str, Any], ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.agent_system_prompt, str) or not self.agent_system_prompt.strip():
            raise ValueError("agent_system_prompt must be a non-empty string")
        if not isinstance(self.public_history, tuple):
            raise TypeError("public_history must be a tuple")
        if self.tool_schemas is not None and not isinstance(self.tool_schemas, tuple):
            raise TypeError("tool_schemas must be a tuple or None")
        if self.tool_schemas is not None and any(
            not isinstance(schema, Mapping) for schema in self.tool_schemas
        ):
            raise TypeError("tool_schemas must contain mappings")


def _project_history_call(call: Any) -> dict[str, Any]:
    if not isinstance(call, Mapping):
        raise TypeError("public history tool calls must be mappings")
    name = call.get("name")
    arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ValueError("public history tool call name must be non-empty")
    if not isinstance(arguments, Mapping):
        raise TypeError("public history tool call arguments must be a mapping")
    projected: dict[str, Any] = {
        "name": name,
        "arguments": _json_clone(dict(arguments)),
    }
    if "result" in call:
        projected["result"] = _json_clone(call["result"])
    return projected


def project_public_history(messages: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Project only user-visible and Producer-visible canonical fields."""

    projected: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("STATE-Bench history messages must be mappings")
        role = message.get("role")
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"non-public history role is forbidden: {role!r}")
        row: dict[str, Any] = {"role": role}
        content = message.get("content")
        if content is not None:
            if not isinstance(content, (str, list, dict)):
                raise TypeError("public history content must be JSON text/object/array or None")
            row["content"] = _json_clone(content)
        calls = message.get("tool_calls")
        if calls is not None:
            if role != "assistant" or not isinstance(calls, (list, tuple)):
                raise ValueError("tool_calls are allowed only on assistant messages")
            row["tool_calls"] = [_project_history_call(call) for call in calls]
        projected.append(row)
    return projected


def project_tool_result(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project one adapter-owned native tool observation."""

    if not isinstance(observation, Mapping):
        raise TypeError("STATE-Bench observation must be a mapping")
    call = project_tool_call(observation)
    if "result" not in observation:
        raise ValueError("STATE-Bench observation is missing result")
    return {**call, "result": _json_clone(observation["result"])}


def project_state_diff(before_state: Any, after_state: Any) -> list[dict[str, Any]]:
    """Return changed paths only, omitting full records removed by a candidate."""

    safe: list[dict[str, Any]] = []
    for change in structural_diff(before_state, after_state):
        projected = {"op": change["op"], "path": change["path"]}
        # A remove operation's full old record can reveal unqueried task state.
        # The path is enough to identify the deleted entity/field.
        if change["op"] != "remove" and "before" in change:
            projected["before"] = _json_clone(change["before"])
        if "after" in change:
            projected["after"] = _json_clone(change["after"])
        safe.append(projected)
    return safe


def build_state_bench_evidence(
    *,
    context: StateBenchEvidenceContext,
    action: StateBenchToolBatch,
    observations: tuple[Mapping[str, Any], ...],
    before_state: Mapping[str, Any],
    after_state: Mapping[str, Any],
    candidate_policy_test_report: Mapping[str, Any] | None = None,
) -> Evidence:
    """Build the only value passed to a fresh Shadow reviewer."""

    if not isinstance(context, StateBenchEvidenceContext):
        raise TypeError("context must be StateBenchEvidenceContext")
    if not isinstance(action, StateBenchToolBatch):
        raise TypeError("action must be StateBenchToolBatch")
    if not isinstance(observations, tuple) or len(observations) != len(action.calls):
        raise ValueError("observations must contain one result per tool call")
    effect = [
        Part.json(
            "candidate_tool_results",
            [project_tool_result(observation) for observation in observations],
        ),
        Part.json("candidate_state_diff", project_state_diff(before_state, after_state)),
    ]
    if candidate_policy_test_report is not None:
        if not isinstance(candidate_policy_test_report, Mapping):
            raise TypeError("candidate_policy_test_report must be a mapping or None")
        effect.append(
            Part.json(
                "candidate_policy_test_report",
                _json_clone(candidate_policy_test_report),
            )
        )
    context_parts = [
        Part.text("agent_system_prompt", context.agent_system_prompt),
        Part.json("public_history", project_public_history(context.public_history)),
    ]
    if context.tool_schemas is not None:
        context_parts.append(
            Part.json(
                "visible_tool_schemas",
                [_json_clone(schema) for schema in context.tool_schemas],
            )
        )
    return Evidence(
        context=tuple(context_parts),
        action=(Part.json("assistant_tool_batch", action.as_json()),),
        effect=tuple(effect),
    )
