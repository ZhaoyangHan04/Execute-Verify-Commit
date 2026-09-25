"""Fail-closed replay of a frozen sequence of semantic review decisions."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Iterable

from ..errors import ReviewInfrastructureError
from ..model import Decision, Evidence


_SHA256 = re.compile(r"[0-9a-f]{64}")


class FrozenReviewMismatchError(ReviewInfrastructureError):
    """The live staged trajectory does not match its frozen review prefix."""


@dataclass(frozen=True)
class FrozenReviewStep:
    evidence_sha256: str
    decision: Decision

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.evidence_sha256) is None:
            raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.decision, Decision):
            raise TypeError("decision must be Decision")


class _FrozenReviewer:
    def __init__(self, step: FrozenReviewStep, ordinal: int) -> None:
        self._step = step
        self._ordinal = ordinal
        self._used = False

    def __call__(self, evidence: Evidence, /) -> Decision:
        if self._used:
            raise FrozenReviewMismatchError("frozen reviewer instance was reused")
        self._used = True
        if not isinstance(evidence, Evidence):
            raise FrozenReviewMismatchError("review input is not Evidence")
        actual = evidence.digest()
        if actual != self._step.evidence_sha256:
            raise FrozenReviewMismatchError(
                "frozen review evidence mismatch at ordinal "
                f"{self._ordinal}: expected={self._step.evidence_sha256}, "
                f"actual={actual}"
            )
        return self._step.decision


class FrozenReviewerFactory:
    """Create one fresh reviewer per frozen B1 decision, in strict order."""

    def __init__(self, steps: Iterable[FrozenReviewStep]) -> None:
        self._steps = tuple(steps)
        if not self._steps:
            raise ValueError("frozen review plan must not be empty")
        if any(not isinstance(step, FrozenReviewStep) for step in self._steps):
            raise TypeError("frozen review plan must contain FrozenReviewStep values")
        if any(not step.decision.accept for step in self._steps[:-1]):
            raise ValueError("only the final frozen review step may reject")
        if self._steps[-1].decision.accept:
            raise ValueError("frozen first-rejection plan must end in rejection")
        self._next = 0
        self._lock = threading.Lock()

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._next

    @property
    def total(self) -> int:
        return len(self._steps)

    def __call__(self) -> _FrozenReviewer:
        with self._lock:
            if self._next >= len(self._steps):
                raise FrozenReviewMismatchError(
                    "live trajectory requested review after frozen first rejection"
                )
            ordinal = self._next
            step = self._steps[ordinal]
            self._next += 1
        return _FrozenReviewer(step, ordinal)

    def assert_complete(self) -> None:
        with self._lock:
            consumed = self._next
        if consumed != len(self._steps):
            raise FrozenReviewMismatchError(
                "live trajectory ended before frozen first rejection: "
                f"consumed={consumed}, expected={len(self._steps)}"
            )
