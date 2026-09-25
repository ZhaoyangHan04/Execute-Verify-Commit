"""Minimal, harness-agnostic Shadow Verifier kernel."""

from .errors import (
    ReviewInfrastructureError,
    SessionPoisonedError,
    SettlementError,
    ShadowError,
)
from .model import (
    Candidate,
    Decision,
    Evidence,
    MAX_DECISION_RATIONALE_CHARS,
    Part,
    Published,
    Rejected,
    StepRecord,
    StepResult,
)
from .protocols import ForkExecutor, Reviewer, ReviewerFactory, SampledReviewerFactory
from .runtime import MAX_VERIFICATION_BUDGET, PROBABILITY_AGGREGATIONS, ShadowVerifier

__all__ = [
    "Candidate",
    "Decision",
    "Evidence",
    "ForkExecutor",
    "MAX_DECISION_RATIONALE_CHARS",
    "MAX_VERIFICATION_BUDGET",
    "PROBABILITY_AGGREGATIONS",
    "Part",
    "Published",
    "Rejected",
    "ReviewInfrastructureError",
    "Reviewer",
    "ReviewerFactory",
    "SampledReviewerFactory",
    "SessionPoisonedError",
    "SettlementError",
    "ShadowError",
    "ShadowVerifier",
    "StepRecord",
    "StepResult",
]
