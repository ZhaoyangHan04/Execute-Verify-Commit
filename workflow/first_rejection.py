"""Single historical rejection intervention; never invokes an online reviewer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from shadow_verifier import Decision
from shadow_verifier.reviewers.frozen import FrozenReviewerFactory, FrozenReviewStep


class FirstRejectionDiagnostic:
    def __init__(self, reference: Path, feedback: str, *, allow_historical_infrastructure_rejection: bool = False) -> None:
        if feedback not in {"generic", "original"}:
            raise ValueError("feedback must be generic or original")
        raw = reference.read_bytes()
        history = json.loads(raw)
        events = history.get("gate_events", history.get("records"))
        if not isinstance(events, list):
            raise ValueError("historical B1 reference has no gate records")
        steps = []
        for event in events:
            decisions = event.get("review_decisions", [])
            if len(decisions) != 1:
                raise ValueError("first-rejection diagnostic requires historical Hard B1")
            d = event["decision"]
            if d["code"] in {"invalid_review", "review_unavailable"} and not allow_historical_infrastructure_rejection:
                raise ValueError("historical first rejection is an infrastructure failure")
            decision = Decision(accept=d["accept"], code=d["code"], rationale=d["rationale"])
            if not decision.accept:
                self.original_decision = decision.as_json()
                self.action = event.get("action", event.get("candidate_action", event.get("batch")))
                if feedback == "generic":
                    decision = Decision(False, "rejected", "Action was not executed. Please continue.")
            steps.append(FrozenReviewStep(event["evidence_sha256"], decision))
            if not decision.accept:
                break
        self.factory = FrozenReviewerFactory(steps)
        self._expected_last = steps[-1]
        self._allow_historical_infrastructure_rejection = allow_historical_infrastructure_rejection
        self.finished = False
        self.metadata = {
            "kind": "single_first_rejection_intervention",
            "feedback": feedback,
            "reference": str(reference.resolve()),
            "reference_sha256": hashlib.sha256(raw).hexdigest(),
            "frozen_gate_count": len(steps),
            "first_rejection_action": self.action,
            "original_decision": self.original_decision,
            "online_reviewer_calls": 0,
        }
        if allow_historical_infrastructure_rejection:
            self.metadata["historical_intervention_kind"] = "infrastructure_rejection"

    def check_record(self, record: Any) -> None:
        # Runtime turns reviewer exceptions into fail-closed decisions. Stop
        # the experiment here, before any mismatch becomes Producer feedback.
        if record.decision.code in {"invalid_review", "review_unavailable"}:
            if not (
                self._allow_historical_infrastructure_rejection
                and record.evidence_sha256 == self._expected_last.evidence_sha256
                and record.decision.as_json() == self._expected_last.decision.as_json()
                and self.factory.consumed == self.factory.total
            ):
                raise RuntimeError("frozen first-rejection evidence mismatch")
        if not record.decision.accept:
            self.factory.assert_complete()
            self.finished = True

    def assert_complete(self) -> None:
        self.factory.assert_complete()
        if not self.finished:
            raise RuntimeError("trajectory did not reach its frozen first rejection")


class IndependentFirstRejection:
    """One live rejection, then verifier-off; no historical pairing claim."""

    is_live = True

    def __init__(self, feedback: str) -> None:
        if feedback not in {"generic", "original"}:
            raise ValueError("feedback must be generic or original")
        self.feedback = feedback
        self.finished = False
        self.first_rejection = None
        self.review_count = 0
        self.metadata = {"kind": "independent_single_first_rejection", "feedback": feedback}

    def wrap_factory(self, factory):
        def fresh():
            if self.finished:
                raise RuntimeError("verifier requested after single intervention")
            reviewer = factory()
            def review(evidence):
                decision = reviewer(evidence)
                if decision.code in {"invalid_review", "review_unavailable"}:
                    raise RuntimeError("independent review infrastructure failure")
                self.review_count += 1
                if not decision.accept:
                    self.first_rejection = {
                        "evidence_sha256": evidence.digest(), "original_decision": decision.as_json(),
                        "review_ordinal": self.review_count - 1,
                    }
                    if self.feedback == "generic":
                        return Decision(False, "rejected", "Action was not executed. Please continue.")
                return decision
            return review
        return fresh

    def check_record(self, record) -> None:
        if record.decision.code in {"invalid_review", "review_unavailable"}:
            raise RuntimeError("independent review infrastructure failure")
        if not record.decision.accept:
            self.finished = True

    def assert_complete(self) -> None:
        # No-rejection trajectories are valid, but not applicable to a
        # conditional first-rejection diagnostic.
        if self.first_rejection is not None and not self.finished:
            raise RuntimeError("first rejection was not settled")

    def as_json(self):
        return {**self.metadata, "applicable": self.finished,
                "review_count": self.review_count, "first_rejection": self.first_rejection}
