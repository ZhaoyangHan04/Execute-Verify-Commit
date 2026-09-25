"""τ² half-duplex orchestration with a batch-level Shadow publication barrier."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any

from shadow_verifier import Published, ShadowVerifier
from shadow_verifier.policy_tests import (
    CandidatePolicyTestRunner,
    PolicyTestSuite,
)
from tau2.orchestrator.orchestrator import Orchestrator

from .environment import Tau2ForkExecutor
from .candidate_execution import Tau2CandidateExecutionProvider
from .evidence import Tau2EvidenceContext
from .model import Tau2ToolBatch
from .trajectory import canonicalize_messages


class ShadowTau2Orchestrator(Orchestrator):
    """Gate each assistant batch containing a mutation as one atomic action."""

    def __init__(
        self,
        *args: Any,
        reviewer_factory,
        verification_budget: int = 1,
        probability_aggregation: str = "vote_majority",
        probability_threshold: float = 0.5,
        probability_early_stop: bool = False,
        include_tool_schemas_in_evidence: bool = True,
        candidate_test_suite: PolicyTestSuite | None = None,
        on_first_rejection: Callable[[], None] | None = None,
        first_rejection_diagnostic: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if type(include_tool_schemas_in_evidence) is not bool:
            raise TypeError("include_tool_schemas_in_evidence must be bool")
        canonical_environment = kwargs["environment"]
        self._judge_tool_schemas = tuple(
            copy.deepcopy(tool.openai_schema)
            for tool in canonical_environment.get_tools()
        )
        self._include_tool_schemas_in_evidence = include_tool_schemas_in_evidence
        self.rejected_call_ids: set[str] = set()
        self.published_call_ids: set[str] = set()
        self.gate_events: list[dict[str, Any]] = []
        self._pending_batch: Tau2ToolBatch | None = None
        self._on_first_rejection = on_first_rejection
        self._first_rejection_seen = False
        self._first_rejection_diagnostic = first_rejection_diagnostic
        self._candidate_policy_tests_enabled = candidate_test_suite is not None
        if candidate_test_suite is not None and not isinstance(
            candidate_test_suite, PolicyTestSuite
        ):
            raise TypeError("candidate_test_suite must be PolicyTestSuite or None")
        candidate_provider = (
            Tau2CandidateExecutionProvider(canonical_environment)
            if candidate_test_suite is not None
            else None
        )
        candidate_runner = (
            CandidatePolicyTestRunner(
                candidate_test_suite,
                candidate_provider.candidate_schema,
            )
            if candidate_test_suite is not None and candidate_provider is not None
            else None
        )
        executor = Tau2ForkExecutor(
            canonical=canonical_environment,
            context_provider=self._context_for,
            candidate_execution_provider=candidate_provider,
            candidate_test_runner=candidate_runner,
        )
        kwargs["environment"] = executor
        super().__init__(*args, **kwargs)
        self.fork_executor = executor
        self.shadow = ShadowVerifier(
            executor,
            reviewer_factory,
            verification_budget=verification_budget,
            probability_aggregation=probability_aggregation,
            probability_threshold=probability_threshold,
            probability_early_stop=probability_early_stop,
        )

    def _context_for(self) -> Tau2EvidenceContext:
        """Build a clean Judge view before the current proposal.

        The global orchestrator trajectory may contain user-tool events that
        are private to the UserSimulator.  The Judge must instead start from
        the official Agent state's role-filtered history.  At this hook that
        history already ends in the current assistant proposal; remove it to
        avoid duplicating the action in Evidence.
        """

        batch = self._pending_batch
        from tau2.agent.base_agent import is_valid_agent_history_message

        producer_history = tuple(
            message
            for message in self.trajectory
            if is_valid_agent_history_message(message)
        )
        if batch is None or not producer_history:
            raise RuntimeError("missing pending tau2 batch while building evidence")
        current = producer_history[-1]
        current_calls = tuple(
            getattr(call, "id", None)
            for call in (getattr(current, "tool_calls", None) or ())
        )
        if current_calls != batch.call_ids:
            raise RuntimeError(
                "current tau2 trajectory proposal does not match pending batch"
            )
        public_prefix = canonicalize_messages(
            producer_history[:-1],
            rejected_call_ids=self.rejected_call_ids,
        )
        return Tau2EvidenceContext(
            policy=self.fork_executor.canonical.get_policy(),
            public_history=tuple(public_prefix),
            tool_schemas=(
                self._judge_tool_schemas
                if self._include_tool_schemas_in_evidence
                else None
            ),
        )

    @staticmethod
    def _batch_call_id(batch: Tau2ToolBatch) -> str:
        payload = json.dumps(
            list(batch.call_ids),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "tau2-batch:" + hashlib.sha256(payload).hexdigest()

    def _execute_tool_calls(self, tool_calls):
        if not tool_calls:
            return []
        if any(call.requestor != "assistant" for call in tool_calls):
            return super()._execute_tool_calls(tool_calls)
        if self._first_rejection_diagnostic is not None and self._first_rejection_diagnostic.finished:
            return super()._execute_tool_calls(tool_calls)

        mutates = any(
            self.fork_executor.canonical._is_mutating_tool(call.name)
            for call in tool_calls
        )
        if not mutates:
            return super()._execute_tool_calls(tool_calls)

        batch = Tau2ToolBatch(calls=tuple(tool_calls))
        if self._pending_batch is not None:
            raise RuntimeError("nested tau2 Shadow batch is not supported")
        self._pending_batch = batch
        try:
            result = self.shadow.step(self._batch_call_id(batch), batch)
        finally:
            self._pending_batch = None
        if self._first_rejection_diagnostic is not None:
            self._first_rejection_diagnostic.check_record(self.shadow.audit_records[-1])
        observations = list(result.observation)
        if isinstance(result, Published):
            self.published_call_ids.update(batch.call_ids)
        else:
            self.rejected_call_ids.update(batch.call_ids)
            if not self._first_rejection_seen:
                self._first_rejection_seen = True
                if self._on_first_rejection is not None:
                    self._on_first_rejection()
        # Match the official orchestrator: every Producer-visible ToolMessage
        # with error=True consumes the same error budget, including bounded
        # synthetic rejection observations.
        self.num_errors += sum(bool(item.error) for item in observations)
        record = self.shadow.audit_records[-1]
        self.gate_events.append(
            {
                "call_id": record.call_id,
                "tool_call_ids": list(batch.call_ids),
                "action": batch.as_json(),
                "outcome": record.outcome,
                "decision": record.decision.as_json(),
                "verification_budget": len(record.review_decisions),
                "accept_votes": sum(
                    decision.accept for decision in record.review_decisions
                ),
                "reject_votes": sum(
                    not decision.accept for decision in record.review_decisions
                ),
                "review_decisions": [
                    {"sample_index": sample_index, **decision.as_json()}
                    for sample_index, decision in enumerate(record.review_decisions)
                ],
                "evidence_sha256": record.evidence_sha256,
                "stage_seconds": record.stage_seconds,
                "review_seconds": record.review_seconds,
                "settle_seconds": record.settle_seconds,
                "total_seconds": record.total_seconds,
                **(
                    {
                        "candidate_policy_test_report": (
                            self.fork_executor.test_reports[-1]
                        )
                    }
                    if self._candidate_policy_tests_enabled
                    else {}
                ),
            }
        )
        return observations

    def canonical_trajectory(self) -> list[Any]:
        return canonicalize_messages(
            self.get_trajectory(),
            rejected_call_ids=self.rejected_call_ids,
        )
