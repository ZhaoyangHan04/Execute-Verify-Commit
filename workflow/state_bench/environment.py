"""Deep-copy stage and exact-publish mechanics for STATE-Bench environments."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from shadow_verifier import Candidate, Decision
from shadow_verifier.policy_tests import CandidatePolicyTestRunner

from .candidate_execution import StateBenchCandidateExecutionProvider
from .evidence import StateBenchEvidenceContext, build_state_bench_evidence
from .model import StateBenchToolBatch


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


@dataclass
class _StateBenchCandidateHandle:
    owner: object
    action: StateBenchToolBatch
    environment: Any | None
    observations: tuple[dict[str, Any], ...]
    before_generation: int
    before_fingerprint: str
    published: bool = False
    discarded: bool = False


class StateBenchForkExecutor:
    """A single-episode executor over one official in-memory environment."""

    def __init__(
        self,
        canonical: Any,
        write_tool_names: Iterable[str],
        context_provider: Callable[[], StateBenchEvidenceContext],
        candidate_execution_provider: StateBenchCandidateExecutionProvider | None = None,
        candidate_test_runner: CandidatePolicyTestRunner | None = None,
    ) -> None:
        if canonical is None or not callable(getattr(canonical, "get_full_snapshot", None)):
            raise TypeError("canonical must be a STATE-Bench environment")
        handlers = getattr(canonical, "tool_handlers", None)
        if not isinstance(handlers, Mapping):
            raise TypeError("canonical.tool_handlers must be a mapping")
        writes = frozenset(write_tool_names)
        if any(not isinstance(name, str) or not name for name in writes):
            raise ValueError("write_tool_names must contain non-empty strings")
        missing = writes - set(handlers)
        if missing:
            raise ValueError(f"write tools missing handlers: {sorted(missing)!r}")
        if not callable(context_provider):
            raise TypeError("context_provider must be callable")
        if (candidate_execution_provider is None) != (candidate_test_runner is None):
            raise ValueError(
                "candidate_execution_provider and candidate_test_runner must be enabled together"
            )

        self._canonical = canonical
        self._write_tool_names = writes
        self._context_provider = context_provider
        self._candidate_execution_provider = candidate_execution_provider
        self._candidate_test_runner = candidate_test_runner
        self._test_reports: list[dict[str, Any]] = []
        self._generation = 0
        self._owner = object()
        self._sequence = itertools.count(1)
        self._active: dict[str, _StateBenchCandidateHandle] = {}
        self._lock = threading.RLock()

    @property
    def canonical(self) -> Any:
        """Return an isolated diagnostic copy, never the mutable live object."""

        with self._lock:
            return copy.deepcopy(self._canonical)

    @property
    def write_tool_names(self) -> frozenset[str]:
        return self._write_tool_names

    @property
    def test_reports(self) -> tuple[dict[str, Any], ...]:
        return tuple(_json_clone(value) for value in self._test_reports)

    def get_full_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot(self._canonical)

    def requires_review(self, action: StateBenchToolBatch) -> bool:
        if not isinstance(action, StateBenchToolBatch):
            raise TypeError("action must be StateBenchToolBatch")
        return bool(set(action.names) & self._write_tool_names)

    def fingerprint(self, action: StateBenchToolBatch, /) -> str:
        if not isinstance(action, StateBenchToolBatch):
            raise TypeError("action must be StateBenchToolBatch")
        return action.fingerprint()

    def execute_direct(self, action: StateBenchToolBatch) -> tuple[dict[str, Any], ...]:
        """Execute a baseline batch or a Shadow read-only batch canonically."""

        if not isinstance(action, StateBenchToolBatch):
            raise TypeError("action must be StateBenchToolBatch")
        with self._lock:
            try:
                return self._execute(self._canonical, action)
            finally:
                # Reads such as customer_support.get_policies mutate private
                # gate state, so every direct interaction advances the CAS
                # generation even when the official snapshot is unchanged.
                self._generation += 1

    def stage(
        self,
        call_id: str,
        action: StateBenchToolBatch,
        /,
    ) -> Candidate[_StateBenchCandidateHandle]:
        if not isinstance(action, StateBenchToolBatch):
            raise TypeError("action must be StateBenchToolBatch")
        if not self.requires_review(action):
            raise ValueError("stage accepts only batches containing a write-capable tool")
        with self._lock:
            context = self._context_provider()
            if not isinstance(context, StateBenchEvidenceContext):
                raise TypeError("context_provider must return StateBenchEvidenceContext")
            before_generation = self._generation
            before_state = self._snapshot(self._canonical)
            before_fingerprint = self._snapshot_fingerprint(before_state)
            candidate_environment = copy.deepcopy(self._canonical)
            observations = self._execute(candidate_environment, action)
            after_state = self._snapshot(candidate_environment)
            test_report = None
            if self._candidate_test_runner is not None:
                assert self._candidate_execution_provider is not None
                candidate_execution = (
                    self._candidate_execution_provider.build_candidate_execution(
                        now=str(getattr(self._canonical, "now", "")),
                        public_history=context.public_history,
                        action=action,
                        observations=observations,
                        before_state=before_state,
                        after_state=after_state,
                    )
                )
                test_report = self._candidate_test_runner.run(candidate_execution)
                self._test_reports.append(_json_clone(test_report))
            evidence = build_state_bench_evidence(
                context=context,
                action=action,
                observations=observations,
                before_state=before_state,
                after_state=after_state,
                candidate_policy_test_report=test_report,
            )

            sequence = next(self._sequence)
            candidate_id = hashlib.sha256(
                f"{call_id}\0{sequence}\0{action.fingerprint()}\0{before_fingerprint}".encode()
            ).hexdigest()
            handle = _StateBenchCandidateHandle(
                owner=self._owner,
                action=action,
                environment=candidate_environment,
                observations=observations,
                before_generation=before_generation,
                before_fingerprint=before_fingerprint,
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
        candidate: Candidate[_StateBenchCandidateHandle],
        /,
    ) -> tuple[dict[str, Any], ...]:
        with self._lock:
            handle = self._require_active(candidate)
            if handle.environment is None:
                raise RuntimeError("candidate environment is unavailable")
            current_fingerprint = self._snapshot_fingerprint(self._snapshot(self._canonical))
            if (
                self._generation != handle.before_generation
                or current_fingerprint != handle.before_fingerprint
            ):
                raise RuntimeError("canonical STATE-Bench environment changed before publish")

            # Publish the exact environment that produced reviewer evidence.
            # Do not replay: preview and policy-gate state is not in official
            # snapshots.  Clear the externally reachable handle afterwards so
            # it cannot be used as a mutable alias to the live environment.
            # Clone the fully executed fork once more at the trust boundary.
            # This is not replay: it preserves every hidden preview/policy
            # field while preventing a caller that retained the staged object
            # from mutating the now-live environment after settlement.
            promoted = copy.deepcopy(handle.environment)
            self._canonical = promoted
            handle.environment = None
            self._generation += 1
            handle.published = True
            self._active.pop(candidate.candidate_id, None)
            return handle.observations

    def rejection(
        self,
        candidate: Candidate[_StateBenchCandidateHandle],
        decision: Decision,
        /,
    ) -> tuple[dict[str, Any], ...]:
        with self._lock:
            handle = self._require_active(candidate)
            if not isinstance(decision, Decision) or decision.accept:
                raise ValueError("rejection requires a rejected Decision")
            error_text = decision.rationale or decision.code
            feedback = {
                "error": f"Shadow verifier rejected this batch: {error_text}",
                "status": "rejected_by_shadow_verifier",
                "code": decision.code,
                "rationale": decision.rationale,
                "environment_changed": False,
            }
            return tuple(
                {
                    "id": call["id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "result": _json_clone(feedback),
                }
                for call in handle.action.as_json()
            )

    def discard(self, candidate: Candidate[_StateBenchCandidateHandle], /) -> None:
        with self._lock:
            handle = self._require_active(candidate)
            handle.environment = None
            handle.discarded = True
            self._active.pop(candidate.candidate_id, None)

    def _execute(
        self,
        environment: Any,
        action: StateBenchToolBatch,
    ) -> tuple[dict[str, Any], ...]:
        observations: list[dict[str, Any]] = []
        for call in action.as_json():
            handler = environment.tool_handlers.get(call["name"])
            if handler is None:
                raise ValueError(f"unknown STATE-Bench domain tool: {call['name']!r}")
            result = handler(_json_clone(call["arguments"]))
            observations.append({**call, "result": _json_clone(result)})
        return tuple(observations)

    def _require_active(
        self,
        candidate: Candidate[_StateBenchCandidateHandle],
    ) -> _StateBenchCandidateHandle:
        if not isinstance(candidate, Candidate):
            raise TypeError("candidate must be shadow_verifier.Candidate")
        handle = candidate.handle
        if not isinstance(handle, _StateBenchCandidateHandle) or handle.owner is not self._owner:
            raise ValueError("candidate does not belong to this executor")
        if self._active.get(candidate.candidate_id) is not handle:
            raise RuntimeError("candidate is not active")
        if handle.published or handle.discarded:
            raise RuntimeError("candidate has already been settled")
        return handle

    @staticmethod
    def _snapshot(environment: Any) -> dict[str, Any]:
        snapshot = environment.get_full_snapshot()
        if not isinstance(snapshot, Mapping):
            raise TypeError("get_full_snapshot() must return a mapping")
        return _json_clone(snapshot)

    @staticmethod
    def _snapshot_fingerprint(snapshot: Mapping[str, Any]) -> str:
        payload = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()
