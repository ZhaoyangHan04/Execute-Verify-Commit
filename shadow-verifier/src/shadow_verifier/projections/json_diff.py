"""Deterministic structural diffs over JSON-compatible values."""

from __future__ import annotations

import copy
from typing import Any


def _pointer(path: str, key: str | int) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def structural_diff(
    before: Any,
    after: Any,
    *,
    _path: str = "",
) -> list[dict[str, Any]]:
    """Return a deterministic JSON-Patch-like delta without unchanged state."""

    if type(before) is not type(after):
        return [
            {
                "op": "replace",
                "path": _path or "/",
                "before": copy.deepcopy(before),
                "after": copy.deepcopy(after),
            }
        ]
    if isinstance(before, dict):
        changes: list[dict[str, Any]] = []
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys - after_keys, key=str):
            changes.append(
                {
                    "op": "remove",
                    "path": _pointer(_path, key),
                    "before": copy.deepcopy(before[key]),
                }
            )
        for key in sorted(after_keys - before_keys, key=str):
            changes.append(
                {
                    "op": "add",
                    "path": _pointer(_path, key),
                    "after": copy.deepcopy(after[key]),
                }
            )
        for key in sorted(before_keys & after_keys, key=str):
            changes.extend(
                structural_diff(
                    before[key],
                    after[key],
                    _path=_pointer(_path, key),
                )
            )
        return changes
    if isinstance(before, list):
        changes = []
        common = min(len(before), len(after))
        for index in range(common):
            changes.extend(
                structural_diff(
                    before[index],
                    after[index],
                    _path=_pointer(_path, index),
                )
            )
        for index in range(len(before) - 1, common - 1, -1):
            changes.append(
                {
                    "op": "remove",
                    "path": _pointer(_path, index),
                    "before": copy.deepcopy(before[index]),
                }
            )
        for index in range(common, len(after)):
            changes.append(
                {
                    "op": "add",
                    "path": _pointer(_path, index),
                    "after": copy.deepcopy(after[index]),
                }
            )
        return changes
    if before != after:
        return [
            {
                "op": "replace",
                "path": _path or "/",
                "before": copy.deepcopy(before),
                "after": copy.deepcopy(after),
            }
        ]
    return []
