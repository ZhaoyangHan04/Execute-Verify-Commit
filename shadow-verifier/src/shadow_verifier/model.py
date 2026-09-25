"""Immutable values shared by the Shadow kernel and workflow adapters."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar


H = TypeVar("H")
O = TypeVar("O")
# A single semantic reviewer remains capped at 2,000 characters.  The wider
# kernel value leaves room for a bounded majority decision to return every
# rejecting sample's rationale without silently dropping dissenting evidence.
MAX_DECISION_RATIONALE_CHARS = 12_000


@dataclass(frozen=True)
class Part:
    """One explicitly labelled, media-typed piece of reviewer evidence."""

    name: str
    media_type: str
    data: bytes

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Part.name must be a non-empty string")
        if not self.media_type or not isinstance(self.media_type, str):
            raise ValueError("Part.media_type must be a non-empty string")
        if not isinstance(self.data, bytes):
            raise TypeError("Part.data must be bytes")

    @classmethod
    def text(cls, name: str, value: str) -> "Part":
        return cls(
            name=name, media_type="text/plain; charset=utf-8", data=value.encode()
        )

    @classmethod
    def json(cls, name: str, value: object) -> "Part":
        data = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(name=name, media_type="application/json", data=data)

    def as_text(self) -> str:
        return self.data.decode("utf-8", errors="strict")


Packet: TypeAlias = tuple[Part, ...]


@dataclass(frozen=True)
class Evidence:
    """The only information visible to a fresh reviewer."""

    context: Packet
    action: Packet
    effect: Packet

    def __post_init__(self) -> None:
        for name, packet in (
            ("context", self.context),
            ("action", self.action),
            ("effect", self.effect),
        ):
            if not isinstance(packet, tuple):
                raise TypeError(f"Evidence.{name} must be a tuple")
            if not all(isinstance(part, Part) for part in packet):
                raise TypeError(f"Evidence.{name} must contain only Part values")
        if not self.action:
            raise ValueError("Evidence.action must not be empty")

    def digest(self) -> str:
        hasher = hashlib.sha256()
        for section_name, parts in (
            ("context", self.context),
            ("action", self.action),
            ("effect", self.effect),
        ):
            hasher.update(section_name.encode("utf-8"))
            hasher.update(b"\0")
            for part in parts:
                for value in (part.name, part.media_type):
                    encoded = value.encode("utf-8")
                    hasher.update(len(encoded).to_bytes(8, "big"))
                    hasher.update(encoded)
                hasher.update(len(part.data).to_bytes(8, "big"))
                hasher.update(part.data)
        return hasher.hexdigest()


@dataclass(frozen=True)
class Decision:
    """A review decision available to settlement and adapter-owned feedback."""

    accept: bool
    code: str
    rationale: str = ""
    # Present only when the reviewer was explicitly asked for a soft label.
    # Settlement still consumes ``accept`` so legacy executors remain unchanged;
    # experiments can aggregate the preserved raw probability separately.
    pass_probability: float | None = None

    def __post_init__(self) -> None:
        if type(self.accept) is not bool:
            raise TypeError("Decision.accept must be exactly bool")
        if (
            not isinstance(self.code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.code) is None
        ):
            raise ValueError("Decision.code must be a non-empty snake_case token")
        if not isinstance(self.rationale, str):
            raise TypeError("Decision.rationale must be a string")
        if len(self.rationale) > MAX_DECISION_RATIONALE_CHARS:
            raise ValueError("Decision.rationale is too long")
        if self.pass_probability is not None:
            if type(self.pass_probability) not in (int, float):
                raise TypeError("Decision.pass_probability must be a number or None")
            probability = float(self.pass_probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Decision.pass_probability must be finite and in [0, 1]")
            object.__setattr__(self, "pass_probability", probability)
        if self.accept and self.code != "accepted":
            raise ValueError("accepted decisions must use code='accepted'")
        if not self.accept and self.code == "accepted":
            raise ValueError("rejected decisions cannot use code='accepted'")

    @classmethod
    def accepted(
        cls,
        rationale: str = "",
        *,
        pass_probability: float | None = None,
    ) -> "Decision":
        return cls(
            accept=True,
            code="accepted",
            rationale=rationale,
            pass_probability=pass_probability,
        )

    @classmethod
    def rejected(
        cls,
        *,
        code: str,
        rationale: str = "",
        pass_probability: float | None = None,
    ) -> "Decision":
        return cls(
            accept=False,
            code=code,
            rationale=rationale,
            pass_probability=pass_probability,
        )

    def as_json(self) -> dict[str, object]:
        """Return a stable audit payload, omitting absent soft-label metadata."""

        payload: dict[str, object] = {
            "accept": self.accept,
            "code": self.code,
            "rationale": self.rationale,
        }
        if self.pass_probability is not None:
            payload["pass_probability"] = self.pass_probability
        return payload


@dataclass(frozen=True)
class Candidate(Generic[H]):
    """Opaque staged effect. Only its Evidence is passed to the reviewer."""

    call_id: str
    candidate_id: str
    handle: H
    evidence: Evidence


@dataclass(frozen=True)
class Published(Generic[O]):
    call_id: str
    observation: O


@dataclass(frozen=True)
class Rejected(Generic[O]):
    call_id: str
    observation: O


StepResult: TypeAlias = Published[O] | Rejected[O]


@dataclass(frozen=True)
class StepRecord:
    """Audit-only record; never serialize it into an agent observation."""

    call_id: str
    candidate_id: str
    evidence_sha256: str
    outcome: Literal["published", "rejected"]
    decision: Decision
    stage_seconds: float
    review_seconds: float
    settle_seconds: float
    total_seconds: float
    # Audit-only individual votes.  For the legacy/default budget this is a
    # one-element tuple containing ``decision`` itself.
    review_decisions: tuple[Decision, ...] = ()
