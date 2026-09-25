"""Dual-trajectory helpers for official τ² evaluation replay."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_call_ids(message: Any) -> tuple[str, ...]:
    calls = _field(message, "tool_calls")
    if calls is None:
        return ()
    ids = tuple(_field(call, "id") for call in calls)
    if not all(isinstance(call_id, str) and call_id for call_id in ids):
        raise ValueError("trajectory contains a tool call without a valid id")
    return ids


def canonicalize_messages(
    messages: Sequence[Any],
    rejected_call_ids: Iterable[str],
) -> tuple[Any, ...]:
    """Remove complete rejected batches from a Producer-private trajectory.

    The returned tuple is suitable for official environment replay: both the
    rejected assistant proposal and every synthetic ToolMessage linked to it
    are absent.  The input objects are never mutated.  Partial batch filtering
    and dangling rejection IDs fail closed instead of silently producing an
    evaluator trajectory with different transaction semantics.
    """

    rejected = set(rejected_call_ids)
    if not all(isinstance(call_id, str) and call_id for call_id in rejected):
        raise ValueError("rejected_call_ids must contain non-empty strings")
    if not rejected:
        return tuple(messages)

    canonical: list[Any] = []
    action_seen: set[str] = set()
    result_seen: set[str] = set()

    for message in messages:
        role = _field(message, "role")
        call_ids = _message_call_ids(message)
        overlap = set(call_ids) & rejected
        if overlap:
            if role != "assistant":
                raise ValueError("only assistant tool batches may be Shadow-rejected")
            if not set(call_ids).issubset(rejected):
                raise ValueError("cannot remove only part of an atomic tool batch")
            action_seen.update(call_ids)
            continue

        nested = _field(message, "tool_messages")
        if nested is not None:
            nested_ids = tuple(_field(item, "id") for item in nested)
            nested_overlap = set(nested_ids) & rejected
            if nested_overlap:
                if not set(nested_ids).issubset(rejected):
                    raise ValueError("cannot partially filter a MultiToolMessage")
                if any(_field(item, "requestor", "assistant") != "assistant" for item in nested):
                    raise ValueError("rejected ToolMessages must belong to assistant calls")
                result_seen.update(nested_ids)
                continue

        if role == "tool":
            message_id = _field(message, "id")
            if message_id in rejected:
                if _field(message, "requestor", "assistant") != "assistant":
                    raise ValueError("rejected ToolMessage must belong to assistant call")
                result_seen.add(message_id)
                continue

        canonical.append(message)

    missing_actions = rejected - action_seen
    missing_results = rejected - result_seen
    if missing_actions or missing_results:
        raise ValueError(
            "rejected call ids are incomplete in producer trajectory: "
            f"missing_actions={sorted(missing_actions)!r}, "
            f"missing_results={sorted(missing_results)!r}"
        )
    return tuple(canonical)
