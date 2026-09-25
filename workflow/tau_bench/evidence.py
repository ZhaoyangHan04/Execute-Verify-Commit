"""Leakage-safe reviewer evidence for original τ-bench actions."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from shadow_verifier import Evidence, Part
from shadow_verifier.projections import structural_diff

from .model import TauBenchToolAction


_MESSAGE_FIELDS = frozenset({"role", "content", "name", "tool_call_id", "tool_calls"})

_ACTION_POLICY_HEADINGS: dict[str, tuple[str, ...]] = {
    "book_reservation": ("Book flight",),
    "cancel_reservation": ("Cancel flight",),
    "send_certificate": ("Refund",),
    "update_reservation_baggages": ("Modify flight",),
    "update_reservation_flights": ("Modify flight",),
    "update_reservation_passengers": ("Modify flight",),
    "cancel_pending_order": ("Cancel pending order",),
    "exchange_delivered_order_items": ("Exchange delivered order",),
    "modify_pending_order_address": ("Modify pending order",),
    "modify_pending_order_items": ("Modify pending order", "Modify items"),
    "modify_pending_order_payment": ("Modify pending order", "Modify payment"),
    "modify_user_address": (),
    "return_delivered_order_items": ("Return delivered order",),
}


def action_policy_focus(policy: str, action_name: str) -> dict[str, Any]:
    """Select verbatim global and action-local excerpts from the official wiki."""

    if action_name not in _ACTION_POLICY_HEADINGS:
        raise ValueError(f"unsupported mutating action for policy focus: {action_name}")
    blocks: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in policy.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            if lines:
                blocks.append((heading, "\n".join(lines).strip()))
            heading = line.lstrip("#").strip()
            lines = [line]
        else:
            lines.append(line)
    if lines:
        blocks.append((heading, "\n".join(lines).strip()))

    wanted = set(_ACTION_POLICY_HEADINGS[action_name])
    excerpts = [text for name, text in blocks if name is None or name in wanted]
    if not excerpts or (wanted and not wanted.issubset({name for name, _ in blocks})):
        raise ValueError("official policy is missing an expected action heading")
    return {
        "action": action_name,
        "source": "verbatim excerpts from domain_policy",
        "excerpts": excerpts,
    }


def project_public_history(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the native fields visible to the producer."""

    projected: list[dict[str, Any]] = []
    for message in messages:
        if type(message) is not dict:
            raise TypeError("τ-bench public history messages must be dictionaries")
        row = {
            key: copy.deepcopy(value)
            for key, value in message.items()
            if key in _MESSAGE_FIELDS and value is not None
        }
        if row.get("role") not in {"user", "assistant", "tool"}:
            raise ValueError("τ-bench public history contains an unsupported role")
        projected.append(row)
    return projected


def build_evidence(
    *,
    policy: str,
    public_history: Sequence[dict[str, Any]],
    tool_schemas: Sequence[Mapping[str, Any]] | None,
    action: TauBenchToolAction,
    observation: str,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    policy_presentation: str = "full",
) -> Evidence:
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("policy must be non-empty")
    if tool_schemas is not None and any(
        not isinstance(schema, Mapping) for schema in tool_schemas
    ):
        raise TypeError("tool_schemas must contain mappings")
    if not isinstance(observation, str):
        raise TypeError("observation must be text")
    if policy_presentation not in {"full", "action_focus_v1"}:
        raise ValueError("unknown policy presentation")
    context_parts = []
    if policy_presentation == "action_focus_v1":
        context_parts.append(
            Part.json(
                "action_policy_focus",
                action_policy_focus(policy, action.name),
            )
        )
    context_parts.extend(
        [
            Part.text("domain_policy", policy),
            Part.json("public_history", project_public_history(public_history)),
        ]
    )
    if tool_schemas is not None:
        context_parts.append(
            Part.json("visible_tool_schemas", copy.deepcopy(list(tool_schemas)))
        )
    return Evidence(
        context=tuple(context_parts),
        action=(Part.json("assistant_tool_call", action.as_json()),),
        effect=(
            Part.json(
                "candidate_tool_result",
                {
                    "content": observation,
                    "error": observation.startswith("Error:"),
                },
            ),
            Part.json(
                "candidate_state_diff",
                structural_diff(before_state, after_state),
            ),
        ),
    )
