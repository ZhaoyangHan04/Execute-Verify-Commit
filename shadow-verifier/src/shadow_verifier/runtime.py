"""The fixed stage-review-publish/discard orchestration."""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from .errors import ReviewInfrastructureError, SessionPoisonedError, SettlementError
from .model import (
    Candidate,
    Decision,
    Evidence,
    Published,
    Rejected,
    StepRecord,
    StepResult,
)
from .protocols import ForkExecutor, ReviewerFactory


A = TypeVar("A")
H = TypeVar("H")
O = TypeVar("O")
MAX_VERIFICATION_BUDGET = 7
PROBABILITY_AGGREGATIONS = frozenset(
    {"vote_majority", "mean_probability", "max_probability"}
)
_EARLY_STOP_REJECT_THRESHOLD = 0.3
_EARLY_STOP_ACCEPT_THRESHOLD = 0.7


@dataclass(frozen=True)
class _Completed(Generic[O]):
    action_fingerprint: str
    result: StepResult[O]


class ShadowVerifier(Generic[A, H, O]):
    """A single-writer pre-publication barrier bound to one environment episode."""

    def __init__(
        self,
        executor: ForkExecutor[A, H, O],
        reviewer_factory: ReviewerFactory,
        *,
        verification_budget: int = 1,
        probability_aggregation: str = "vote_majority",
        probability_threshold: float = 0.5,
        probability_early_stop: bool = False,
        parallel_reviews: bool = False,
    ) -> None:
        if type(verification_budget) is not int:
            raise TypeError("verification_budget must be int")
        if (
            verification_budget <= 0
            or verification_budget > MAX_VERIFICATION_BUDGET
            or verification_budget % 2 == 0
        ):
            raise ValueError(
                "verification_budget must be an odd integer in "
                f"[1, {MAX_VERIFICATION_BUDGET}]"
            )
        if verification_budget > 1 and not callable(
            getattr(reviewer_factory, "for_sample", None)
        ):
            raise TypeError(
                "verification_budget > 1 requires reviewer_factory.for_sample()"
            )
        if probability_aggregation not in PROBABILITY_AGGREGATIONS:
            raise ValueError(
                "probability_aggregation must be one of: "
                + ", ".join(sorted(PROBABILITY_AGGREGATIONS))
            )
        if type(probability_threshold) not in (int, float):
            raise TypeError("probability_threshold must be a number")
        probability_threshold = float(probability_threshold)
        if (
            not math.isfinite(probability_threshold)
            or not 0.5 <= probability_threshold <= 1.0
        ):
            raise ValueError("probability_threshold must be finite and in [0.5, 1]")
        if (
            probability_aggregation != "mean_probability"
            and probability_threshold != 0.5
        ):
            raise ValueError(
                "non-mean probability aggregation requires probability_threshold=0.5"
            )
        if type(probability_early_stop) is not bool:
            raise TypeError("probability_early_stop must be bool")
        if probability_early_stop and verification_budget == 1:
            raise ValueError("probability_early_stop requires verification_budget > 1")
        if probability_early_stop and probability_aggregation != "mean_probability":
            raise ValueError(
                "probability_early_stop requires mean_probability aggregation"
            )
        if type(parallel_reviews) is not bool:
            raise TypeError("parallel_reviews must be bool")
        if parallel_reviews and probability_early_stop:
            raise ValueError("parallel_reviews cannot be combined with early stop")
        self._executor = executor
        self._reviewer_factory = reviewer_factory
        self._verification_budget = verification_budget
        self._probability_aggregation = probability_aggregation
        self._probability_threshold = probability_threshold
        self._probability_early_stop = probability_early_stop
        self._parallel_reviews = parallel_reviews
        self._completed: dict[str, _Completed[O]] = {}
        self._records: list[StepRecord] = []
        self._poisoned = False
        self._lock = threading.RLock()

    @property
    def audit_records(self) -> tuple[StepRecord, ...]:
        return tuple(self._records)

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def verification_budget(self) -> int:
        return self._verification_budget

    @property
    def probability_aggregation(self) -> str:
        return self._probability_aggregation

    @property
    def probability_threshold(self) -> float:
        return self._probability_threshold

    @property
    def probability_early_stop(self) -> bool:
        return self._probability_early_stop

    @property
    def parallel_reviews(self) -> bool:
        return self._parallel_reviews

    def step(self, call_id: str, action: A, /) -> StepResult[O]:
        """Stage once, review in a fresh context, then publish or discard.

        ``call_id`` is episode-scoped. Repeating a completed ID is a transport
        retry and returns the cached result without staging the action again.
        Callers must never reuse an ID for a semantically new action.
        """
        if not call_id or not call_id.strip():
            raise ValueError("call_id must be non-empty")

        with self._lock:
            if self._poisoned:
                raise SessionPoisonedError(
                    "Shadow session is poisoned by an uncertain prior settlement"
                )
            action_fingerprint = self._executor.fingerprint(action)
            if not isinstance(action_fingerprint, str) or not action_fingerprint:
                raise ValueError("executor fingerprint must be a non-empty string")
            cached = self._completed.get(call_id)
            if cached is not None:
                if cached.action_fingerprint != action_fingerprint:
                    raise ValueError(
                        "call_id was reused for a different action: "
                        f"call_id={call_id!r}"
                    )
                return cached.result
            return self._run_new_step(call_id, action, action_fingerprint)

    def _run_new_step(
        self,
        call_id: str,
        action: A,
        action_fingerprint: str,
    ) -> StepResult[O]:
        started = time.perf_counter()
        stage_started = started
        try:
            candidate = self._executor.stage(call_id, action)
        except Exception as exc:
            self._poisoned = True
            raise SettlementError(
                f"stage outcome is uncertain for call_id={call_id!r}; "
                "the episode cannot continue"
            ) from exc
        except BaseException:
            self._poisoned = True
            raise
        stage_seconds = time.perf_counter() - stage_started

        try:
            if not isinstance(candidate, Candidate):
                raise TypeError("executor.stage must return Candidate")
            if candidate.call_id != call_id:
                raise ValueError(
                    "executor returned a candidate for a different call_id: "
                    f"expected={call_id!r}, actual={candidate.call_id!r}"
                )
            if not candidate.candidate_id:
                raise ValueError("candidate_id must be non-empty")
            evidence_sha256 = candidate.evidence.digest()
        except Exception as exc:
            if isinstance(candidate, Candidate):
                self._safe_discard_or_poison(candidate, call_id)
            self._poisoned = True
            raise SettlementError(
                f"invalid staged candidate for call_id={call_id!r}"
            ) from exc

        review_started = time.perf_counter()
        try:
            review_decisions = self._review_votes(candidate.evidence)
            decision = self._aggregate_review_votes(review_decisions)
        except BaseException:
            self._safe_discard_or_poison(candidate, call_id)
            raise
        review_seconds = time.perf_counter() - review_started

        settle_started = time.perf_counter()
        if decision.accept:
            try:
                observation = self._executor.publish(candidate)
            except Exception as exc:
                self._poisoned = True
                raise SettlementError(
                    f"publish outcome is uncertain for call_id={call_id!r}; "
                    "the episode cannot continue"
                ) from exc
            result: StepResult[O] = Published(
                call_id=call_id,
                observation=observation,
            )
            outcome: Literal["published", "rejected"] = "published"
        else:
            try:
                observation = self._executor.rejection(candidate, decision)
            except Exception:
                self._safe_discard_or_poison(candidate, call_id)
                raise
            except BaseException:
                self._safe_discard_or_poison(candidate, call_id)
                raise
            self._safe_discard_or_poison(candidate, call_id)
            result = Rejected(call_id=call_id, observation=observation)
            outcome = "rejected"
        settle_seconds = time.perf_counter() - settle_started

        record = StepRecord(
            call_id=call_id,
            candidate_id=candidate.candidate_id,
            evidence_sha256=evidence_sha256,
            outcome=outcome,
            decision=decision,
            stage_seconds=stage_seconds,
            review_seconds=review_seconds,
            settle_seconds=settle_seconds,
            total_seconds=time.perf_counter() - started,
            review_decisions=review_decisions,
        )
        self._records.append(record)
        self._completed[call_id] = _Completed(
            action_fingerprint=action_fingerprint,
            result=result,
        )
        return result

    def _review_votes(self, evidence: Evidence) -> tuple[Decision, ...]:
        if self._parallel_reviews and self._verification_budget > 1:
            with ThreadPoolExecutor(max_workers=self._verification_budget) as executor:
                return tuple(
                    executor.map(
                        lambda sample_index: self._review_vote(evidence, sample_index),
                        range(self._verification_budget),
                    )
                )

        decisions: list[Decision] = []
        for sample_index in range(self._verification_budget):
            decision = self._review_vote(evidence, sample_index)
            decisions.append(decision)
            if self._probability_early_stop and self._is_extreme_probability(decision):
                break
        return tuple(decisions)

    def _review_vote(self, evidence: Evidence, sample_index: int) -> Decision:
        sampled_factory = getattr(self._reviewer_factory, "for_sample", None)
        try:
            reviewer = (
                self._reviewer_factory()
                if self._verification_budget == 1
                else sampled_factory(sample_index)
            )
            decision = reviewer(evidence)
            self._validate_decision(decision)
        except ReviewInfrastructureError:
            raise
        except (TypeError, ValueError) as exc:
            decision = Decision.rejected(
                code="invalid_review",
                rationale=type(exc).__name__,
            )
        except Exception as exc:
            decision = Decision.rejected(
                code="review_unavailable",
                rationale=type(exc).__name__,
            )
        return decision

    def _aggregate_review_votes(
        self,
        decisions: tuple[Decision, ...],
    ) -> Decision:
        if self._probability_early_stop:
            for decision in decisions:
                if self._is_extreme_probability(decision):
                    return decision
        if len(decisions) != self._verification_budget:
            raise ValueError("review vote count does not match verification_budget")
        if self._probability_aggregation == "mean_probability":
            return self._aggregate_mean_probability(decisions)
        if self._probability_aggregation == "max_probability":
            return self._aggregate_max_probability(decisions)
        if self._verification_budget == 1:
            return decisions[0]
        accept_votes = sum(decision.accept for decision in decisions)
        reject_votes = len(decisions) - accept_votes
        if accept_votes > len(decisions) // 2:
            return Decision.accepted(
                f"majority accepted: accept_votes={accept_votes}, "
                f"reject_votes={reject_votes}, "
                f"verification_budget={len(decisions)}"
            )
        reasons = [
            (
                f"rejection[{sample_index}] code={decision.code}: "
                f"{decision.rationale or '<empty rationale>'}"
            )
            for sample_index, decision in enumerate(decisions)
            if not decision.accept
        ]
        rationale = "\n".join(
            [
                (
                    f"majority rejected: accept_votes={accept_votes}, "
                    f"reject_votes={reject_votes}, "
                    f"verification_budget={len(decisions)}"
                ),
                *reasons,
            ]
        )
        return Decision.rejected(
            code="majority_rejected",
            rationale=rationale,
        )

    def _aggregate_mean_probability(
        self,
        decisions: tuple[Decision, ...],
    ) -> Decision:
        probabilities = self._probabilities(decisions)
        mean_probability = sum(probabilities) / len(probabilities)
        if mean_probability > self._probability_threshold:
            return decisions[max(range(len(decisions)), key=probabilities.__getitem__)]
        representative = decisions[
            min(range(len(decisions)), key=probabilities.__getitem__)
        ]
        if not representative.accept:
            return representative
        # High thresholds can reject a panel even when every p>0.5 vote is a
        # hard accept. Preserve the model's rationale verbatim and keep the
        # probability verifier-side; only normalize the final rejection code.
        return Decision.rejected(
            code="insufficient_evidence",
            rationale=representative.rationale,
            pass_probability=representative.pass_probability,
        )

    def _aggregate_max_probability(
        self,
        decisions: tuple[Decision, ...],
    ) -> Decision:
        probabilities = self._probabilities(decisions)
        return decisions[max(range(len(decisions)), key=probabilities.__getitem__)]

    @staticmethod
    def _probabilities(decisions: tuple[Decision, ...]) -> tuple[float, ...]:
        values = tuple(decision.pass_probability for decision in decisions)
        if any(value is None for value in values):
            raise ValueError(
                "probability aggregation requires pass_probability on every vote"
            )
        return tuple(float(value) for value in values if value is not None)

    @staticmethod
    def _is_extreme_probability(decision: Decision) -> bool:
        probability = decision.pass_probability
        return probability is not None and (
            probability < _EARLY_STOP_REJECT_THRESHOLD
            or probability > _EARLY_STOP_ACCEPT_THRESHOLD
        )

    @staticmethod
    def _validate_decision(decision: object) -> None:
        if not isinstance(decision, Decision):
            raise TypeError("reviewer must return Decision")
        # Re-check even though the dataclass validates construction: a model
        # adapter or malicious object can bypass __init__.
        if type(decision.accept) is not bool:
            raise TypeError("Decision.accept must be exactly bool")
        Decision(
            accept=decision.accept,
            code=decision.code,
            rationale=decision.rationale,
            pass_probability=decision.pass_probability,
        )

    def _safe_discard_or_poison(
        self,
        candidate: Candidate[H],
        call_id: str,
    ) -> None:
        try:
            self._executor.discard(candidate)
        except Exception as exc:
            self._poisoned = True
            raise SettlementError(
                f"candidate cleanup failed for call_id={call_id!r}; "
                "the episode cannot continue"
            ) from exc
