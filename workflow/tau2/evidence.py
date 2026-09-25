"""Leakage-safe evidence projection for the τ² workflow.

Only the domain policy, already-public trajectory, current tool batch, its
candidate results, and the candidate state delta can enter ``Evidence``.
There is intentionally no parameter for a benchmark task, scenario script,
evaluation criterion, reference action, or reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from shadow_verifier import Evidence, Part
from shadow_verifier.projections import structural_diff

from .model import Tau2ToolBatch, project_tool_call


@dataclass(frozen=True)
class Tau2EvidenceContext:
    """The complete online context allowlist for one fresh Judge call."""

    policy: str
    public_history: tuple[Any, ...]
    tool_schemas: tuple[Mapping[str, Any], ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, str) or not self.policy.strip():
            raise ValueError("Tau2EvidenceContext.policy must be non-empty")
        if not isinstance(self.public_history, tuple):
            raise TypeError("Tau2EvidenceContext.public_history must be a tuple")
        if self.tool_schemas is not None and not isinstance(self.tool_schemas, tuple):
            raise TypeError("Tau2EvidenceContext.tool_schemas must be a tuple or None")
        if self.tool_schemas is not None and any(
            not isinstance(schema, Mapping) for schema in self.tool_schemas
        ):
            raise TypeError("Tau2EvidenceContext.tool_schemas must contain mappings")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_clone(value: Any) -> Any:
    """Detach a JSON value from mutable benchmark objects."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return json.loads(encoded)


def _project_history_call(call: Any) -> dict[str, Any]:
    requestor = _field(call, "requestor", "assistant")
    if requestor != "assistant":
        raise ValueError(
            "Judge public history must not contain UserSimulator-private "
            f"tool calls: requestor={requestor!r}"
        )
    return project_tool_call(call)


def project_tool_result(message: Any) -> dict[str, Any]:
    """Allowlist a native ToolMessage and discard all provider metadata."""

    message_id = _field(message, "id")
    requestor = _field(message, "requestor", "assistant")
    content = _field(message, "content")
    error = _field(message, "error", False)
    if not isinstance(message_id, str) or not message_id:
        raise ValueError("tau2 tool result id must be a non-empty string")
    if requestor != "assistant":
        raise ValueError(
            "Judge evidence must not contain UserSimulator-private tool "
            f"results: requestor={requestor!r}"
        )
    if content is not None and not isinstance(content, str):
        raise TypeError("tau2 tool result content must be a string or None")
    if type(error) is not bool:
        raise TypeError("tau2 tool result error must be bool")
    return {
        "id": message_id,
        "requestor": requestor,
        "content": content,
        "error": error,
    }


def project_public_history(messages: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Project only externally visible message fields.

    ``raw_data``, usage, cost, timestamps, provider reasoning, and arbitrary
    benchmark fields are ignored even if present on the native object.
    """

    projected: list[dict[str, Any]] = []
    for message in messages:
        role = _field(message, "role")
        nested_results = _field(message, "tool_messages")
        if nested_results is not None:
            if role != "tool" or not isinstance(nested_results, (tuple, list)):
                raise ValueError("invalid public MultiToolMessage")
            projected.extend(project_tool_result(item) for item in nested_results)
            continue
        if role == "tool":
            projected.append(project_tool_result(message))
            continue
        if role not in ("assistant", "user"):
            raise ValueError(f"non-public role is forbidden in Judge history: {role!r}")
        row: dict[str, Any] = {"role": role}
        content = _field(message, "content")
        if content is not None:
            if not isinstance(content, str):
                raise TypeError("public message content must be a string or None")
            row["content"] = content
        calls = _field(message, "tool_calls")
        if calls is not None:
            if not isinstance(calls, (tuple, list)):
                raise TypeError("public message tool_calls must be a sequence")
            row["tool_calls"] = [_project_history_call(call) for call in calls]
        projected.append(row)
    return projected


def _dump_toolkit_db(toolkit: Any) -> Any:
    if toolkit is None:
        return None
    db = getattr(toolkit, "db", None)
    if db is None:
        return None
    dumper = getattr(db, "model_dump", None)
    if not callable(dumper):
        raise TypeError("tau2 toolkit db must provide model_dump()")
    try:
        value = dumper(mode="json")
    except TypeError:
        value = dumper()
    return _json_clone(value)


def snapshot_environment(environment: Any) -> dict[str, Any]:
    """Capture only the two official τ² database projections."""

    return {
        "agent_db": _dump_toolkit_db(getattr(environment, "tools", None)),
        "user_db": _dump_toolkit_db(getattr(environment, "user_tools", None)),
    }


def build_tau2_evidence(
    *,
    context: Tau2EvidenceContext,
    action: Tau2ToolBatch,
    results: tuple[Any, ...],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    candidate_policy_test_report: Mapping[str, Any] | None = None,
) -> Evidence:
    """Build the complete one-shot Judge view from the fixed allowlist."""

    if not isinstance(context, Tau2EvidenceContext):
        raise TypeError("context_provider must return Tau2EvidenceContext")
    if not isinstance(action, Tau2ToolBatch):
        raise TypeError("action must be Tau2ToolBatch")
    if not isinstance(results, tuple) or len(results) != len(action.calls):
        raise ValueError("results must contain one ToolMessage per batch call")
    effect = [
        Part.json(
            "candidate_tool_results",
            [project_tool_result(result) for result in results],
        ),
        Part.json(
            "candidate_state_diff",
            structural_diff(before_state, after_state),
        ),
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
        Part.text("domain_policy", context.policy),
        Part.json(
            "public_history",
            project_public_history(context.public_history),
        ),
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
