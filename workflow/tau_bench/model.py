"""Small immutable projection of one native τ-bench tool call."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TauBenchToolAction:
    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if type(self.arguments) is not dict:
            raise TypeError("arguments must be a dictionary")
        json.dumps(self.as_json(), ensure_ascii=False, sort_keys=True)

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
