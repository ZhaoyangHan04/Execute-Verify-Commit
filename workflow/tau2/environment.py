"""Fork/execute/promote mechanics for official τ² environments."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Callable

from shadow_verifier import Candidate, Decision
from shadow_verifier.policy_tests import CandidatePolicyTestRunner

from .candidate_execution import Tau2CandidateExecutionProvider
from .evidence import (
    Tau2EvidenceContext,
    build_tau2_evidence,
    snapshot_environment,
)
from .model import Tau2ToolBatch


@dataclass
class _Tau2CandidateHandle:
    owner: object
    action: Tau2ToolBatch
    environment: Any | None
    observations: tuple[Any, ...]
    before_fingerprint: str
    before_generation: int
    published: bool = False
    discarded: bool = False


class Tau2ForkExecutor:
    """Implement the kernel ``ForkExecutor`` protocol for one τ² episode.

    The wrapper owns the canonical environment reference.  ``stage`` deep
    copies the whole environment, executes the complete native batch exactly
    once on that copy, and builds leakage-safe evidence.  ``publish`` swaps in
    that exact copy after a generation-and-state compare-and-swap check.
    """

    def __init__(
        self,
        canonical: Any,
        context_provider: Callable[[], Tau2EvidenceContext],
        candidate_execution_provider: Tau2CandidateExecutionProvider | None = None,
        candidate_test_runner: CandidatePolicyTestRunner | None = None,
    ) -> None:
        if canonical is None:
            raise ValueError("canonical tau2 environment is required")
        if not callable(context_provider):
            raise TypeError("context_provider must be callable")
        if (candidate_execution_provider is None) != (candidate_test_runner is None):
            raise ValueError(
                "candidate_execution_provider and candidate_test_runner must be enabled together"
            )
        self._canonical = canonical
        self._context_provider = context_provider
        self._candidate_execution_provider = candidate_execution_provider
        self._candidate_test_runner = candidate_test_runner
        self._test_reports: list[dict[str, Any]] = []
        self._generation = 0
        self._owner = object()
        self._sequence = itertools.count(1)
        self._active: dict[str, _Tau2CandidateHandle] = {}

    @property
    def test_reports(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(json.dumps(value)) for value in self._test_reports)

    @property
    def canonical(self) -> Any:
        return self._canonical

    @canonical.setter
    def canonical(self, value: Any) -> None:
        if value is None:
            raise ValueError("canonical tau2 environment is required")
        self._canonical = value
        self._generation += 1

    def __getattr__(self, name: str) -> Any:
        """Preserve the official Environment surface for the Orchestrator.

        Python only calls this method after normal executor attributes fail,
        so core methods such as ``stage`` cannot be shadowed by a benchmark
        environment method with the same name.
        """

        return getattr(self._canonical, name)

    def set_state(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate official episode initialization and advance the CAS version."""

        try:
            return self._canonical.set_state(*args, **kwargs)
        finally:
            # A failing initializer may have applied only a prefix of its
            # actions.  Version it as changed so no older candidate can win.
            self._generation += 1

    def get_response(self, message: Any) -> Any:
        """Execute a workflow-bypassed read/user call on canonical state."""

        try:
            return self._canonical.get_response(message)
        finally:
            # Version every direct environment interaction.  This is
            # conservative for reads and also covers a tool that fails after a
            # partial mutation.
            self._generation += 1

    def fingerprint(self, action: Tau2ToolBatch, /) -> str:
        if not isinstance(action, Tau2ToolBatch):
            raise TypeError("Tau2ForkExecutor action must be Tau2ToolBatch")
        return action.fingerprint()

    def stage(
        self,
        call_id: str,
        action: Tau2ToolBatch,
        /,
    ) -> Candidate[_Tau2CandidateHandle]:
        if not isinstance(action, Tau2ToolBatch):
            raise TypeError("Tau2ForkExecutor action must be Tau2ToolBatch")
        context = self._context_provider()
        if not isinstance(context, Tau2EvidenceContext):
            raise TypeError("context_provider must return Tau2EvidenceContext")

        before_generation = self._generation
        before_fingerprint = self._environment_fingerprint(self._canonical)
        before_state = snapshot_environment(self._canonical)
        candidate_environment = copy.deepcopy(self._canonical)

        observations: list[Any] = []
        for tool_call in action.calls:
            observations.append(candidate_environment.get_response(tool_call))
        # τ² domains may maintain cross-toolkit derived state in sync_tools()
        # (Telecom mirrors assistant-side line state into the user device
        # surroundings).  Review and promote the post-batch fixed point, not
        # the state immediately after only the native tool implementations.
        candidate_environment.sync_tools()
        native_observations = tuple(observations)
        after_state = snapshot_environment(candidate_environment)
        test_report = None
        if self._candidate_test_runner is not None:
            assert self._candidate_execution_provider is not None
            candidate_execution = (
                self._candidate_execution_provider.build_candidate_execution(
                    public_history=context.public_history,
                    action=action,
                    results=native_observations,
                    before_state=before_state,
                    after_state=after_state,
                )
            )
            test_report = self._candidate_test_runner.run(candidate_execution)
            self._test_reports.append(json.loads(json.dumps(test_report)))
        evidence = build_tau2_evidence(
            context=context,
            action=action,
            results=native_observations,
            before_state=before_state,
            after_state=after_state,
            candidate_policy_test_report=test_report,
        )

        sequence = next(self._sequence)
        candidate_id = hashlib.sha256(
            (
                f"{call_id}\0{sequence}\0{action.fingerprint()}\0"
                f"{before_fingerprint}"
            ).encode("utf-8")
        ).hexdigest()
        handle = _Tau2CandidateHandle(
            owner=self._owner,
            action=action,
            environment=candidate_environment,
            observations=native_observations,
            before_fingerprint=before_fingerprint,
            before_generation=before_generation,
        )
        self._active[candidate_id] = handle
        return Candidate(
            call_id=call_id,
            candidate_id=candidate_id,
            handle=handle,
            evidence=evidence,
        )

    def publish(
        self,
        candidate: Candidate[_Tau2CandidateHandle],
        /,
    ) -> tuple[Any, ...]:
        handle = self._require_active(candidate)
        if handle.environment is None:
            raise RuntimeError("candidate environment is no longer available")
        current_fingerprint = self._environment_fingerprint(self._canonical)
        if (
            self._generation != handle.before_generation
            or current_fingerprint != handle.before_fingerprint
        ):
            raise RuntimeError("canonical tau2 environment changed before publish")

        # Promote the exact environment that produced the reviewed evidence and
        # native observations.  Never replay the batch on canonical.
        self._canonical = handle.environment
        self._generation += 1
        handle.published = True
        self._active.pop(candidate.candidate_id, None)
        return handle.observations

    def rejection(
        self,
        candidate: Candidate[_Tau2CandidateHandle],
        decision: Decision,
        /,
    ) -> tuple[Any, ...]:
        handle = self._require_active(candidate)
        if not isinstance(decision, Decision) or decision.accept:
            raise ValueError("rejection requires a rejected Decision")
        text = json.dumps(
            {
                "code": decision.code,
                "environment_changed": False,
                "rationale": decision.rationale,
                "status": "rejected_by_shadow_verifier",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        rejected: list[Any] = []
        for call_id, native_result in zip(
            handle.action.call_ids,
            handle.observations,
            strict=True,
        ):
            message_type = type(native_result)
            rejected.append(
                message_type(
                    id=call_id,
                    role="tool",
                    requestor="assistant",
                    error=True,
                    content=text,
                )
            )
        return tuple(rejected)

    def discard(
        self,
        candidate: Candidate[_Tau2CandidateHandle],
        /,
    ) -> None:
        handle = self._require_active(candidate)
        handle.environment = None
        handle.discarded = True
        self._active.pop(candidate.candidate_id, None)

    def _require_active(
        self,
        candidate: Candidate[_Tau2CandidateHandle],
    ) -> _Tau2CandidateHandle:
        if not isinstance(candidate, Candidate):
            raise TypeError("candidate must be shadow_verifier.Candidate")
        handle = candidate.handle
        if (
            not isinstance(handle, _Tau2CandidateHandle)
            or handle.owner is not self._owner
        ):
            raise ValueError("candidate does not belong to this Tau2ForkExecutor")
        if self._active.get(candidate.candidate_id) is not handle:
            raise RuntimeError("candidate is not active")
        if handle.published or handle.discarded:
            raise RuntimeError("candidate has already been settled")
        return handle

    @staticmethod
    def _environment_fingerprint(environment: Any) -> str:
        state = snapshot_environment(environment)
        hashes: dict[str, Any] = {}
        for method_name in ("get_db_hash", "get_user_db_hash"):
            method = getattr(environment, method_name, None)
            if callable(method):
                hashes[method_name] = method()
        payload = json.dumps(
            {"state": state, "official_hashes": hashes},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
