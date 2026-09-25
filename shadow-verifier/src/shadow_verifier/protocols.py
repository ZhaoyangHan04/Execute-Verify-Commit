"""Internal service-provider interfaces used by the fixed core workflow."""

from __future__ import annotations

from typing import Protocol, TypeVar

from .model import Candidate, Decision, Evidence


A = TypeVar("A")
H = TypeVar("H")
O = TypeVar("O")


class Reviewer(Protocol):
    """A one-shot reviewer created with no history from previous steps."""

    def __call__(self, evidence: Evidence, /) -> Decision: ...


class ReviewerFactory(Protocol):
    """Creates one fresh Reviewer for one staged candidate."""

    def __call__(self) -> Reviewer: ...


class SampledReviewerFactory(ReviewerFactory, Protocol):
    """Creates deterministic fresh reviewers for indexed vote samples."""

    def for_sample(self, sample_index: int, /) -> Reviewer: ...


class ForkExecutor(Protocol[A, H, O]):
    """Dataset-owned mechanics; semantic settlement remains in the core."""

    def fingerprint(self, action: A, /) -> str:
        """Return a stable episode-local identity for idempotency checks."""

    def stage(self, call_id: str, action: A, /) -> Candidate[H]:
        """Execute the exact action in isolation and return an opaque candidate."""

    def publish(self, candidate: Candidate[H], /) -> O:
        """Atomically promote this exact candidate and return its native observation."""

    def rejection(self, candidate: Candidate[H], decision: Decision, /) -> O:
        """Map the rejection decision to a native synthetic observation."""

    def discard(self, candidate: Candidate[H], /) -> None:
        """Destroy an unpublished candidate without changing canonical state."""
