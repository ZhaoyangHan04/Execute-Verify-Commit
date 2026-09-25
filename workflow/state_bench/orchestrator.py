"""One STATE-Bench loop shared by direct baseline and Shadow execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from shadow_verifier import Published, ShadowVerifier, StepRecord
from shadow_verifier.policy_tests import (
    CandidatePolicyTestRunner,
    PolicyTestSuite,
)
from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse, BaseAgent
from state_bench.schemas import StateDiff, TokenUsage, Trajectory
from state_bench.scoring import compute_efficiency

from .candidate_execution import StateBenchCandidateExecutionProvider
from .environment import StateBenchForkExecutor
from .evidence import StateBenchEvidenceContext
from .history import StateBenchHistories
from .model import StateBenchToolBatch, StateBenchToolCall
from .runtime_context import build_sanitized_runtime_context

Mode = Literal["baseline", "shadow"]
_REVIEW_INFRASTRUCTURE_CODES = frozenset({"invalid_review", "review_unavailable"})


class StateBenchReviewInfrastructureError(RuntimeError):
    """A staged action was discarded because its review was not trustworthy."""

    def __init__(
        self,
        review_event: Mapping[str, Any],
        *,
        prior_review_events: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        code = review_event.get("decision", {}).get("code", "unknown")
        super().__init__(f"STATE-Bench reviewer infrastructure failure: {code}")
        self.review_event = copy.deepcopy(dict(review_event))
        self.review_events = tuple(
            copy.deepcopy(dict(value)) for value in (*prior_review_events, review_event)
        )


@dataclass(frozen=True)
class StateBenchRun:
    """Result surfaces kept separate from official task ground truth."""

    mode: Mode
    trajectory: Trajectory
    producer_history: tuple[dict[str, Any], ...]
    public_history: tuple[dict[str, Any], ...]
    audit_records: tuple[StepRecord, ...]
    review_events: tuple[dict[str, Any], ...]
    final_snapshot: dict[str, Any]
    execution_metrics: Mapping[str, Any]


def _normalize_response(response: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(response, AgentTurnResponse):
        text = response.text
        raw_calls = response.tool_calls
    elif isinstance(response, Mapping):
        text = str(response.get("text", "") or "")
        raw_calls = response.get("tool_calls", []) or []
    else:
        raise TypeError(
            "generate_next_turn() must return AgentTurnResponse or a mapping"
        )

    calls: list[dict[str, Any]] = []
    for raw in raw_calls:
        if isinstance(raw, AgentToolCallRequest):
            name = raw.name
            arguments = raw.arguments
        elif isinstance(raw, Mapping):
            name = raw.get("name")
            arguments = raw.get("arguments", {})
        else:
            raise TypeError("tool_calls must contain AgentToolCallRequest or mappings")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call request is missing a non-empty name")
        if not isinstance(arguments, Mapping):
            raise TypeError(f"tool call {name!r} arguments must be a mapping")
        calls.append({"name": name, "arguments": dict(arguments)})
    return text, calls


def _review_event(
    *,
    record: StepRecord,
    batch: StateBenchToolBatch,
    observations: tuple[Mapping[str, Any], ...],
    producer_observation_enqueued: bool,
    candidate_policy_test_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete, oracle-free audit record for one reviewed batch."""

    event = {
        "call_id": record.call_id,
        "candidate_id": record.candidate_id,
        "evidence_sha256": record.evidence_sha256,
        "outcome": record.outcome,
        "decision": record.decision.as_json(),
        "verification_budget": len(record.review_decisions),
        "accept_votes": sum(decision.accept for decision in record.review_decisions),
        "reject_votes": sum(
            not decision.accept for decision in record.review_decisions
        ),
        "review_decisions": [
            {"sample_index": sample_index, **decision.as_json()}
            for sample_index, decision in enumerate(record.review_decisions)
        ],
        "candidate_action": copy.deepcopy(batch.as_json()),
        "settlement_observation": copy.deepcopy(list(observations)),
        "producer_observation_enqueued": producer_observation_enqueued,
        "timings_seconds": {
            "stage": record.stage_seconds,
            "review": record.review_seconds,
            "settle": record.settle_seconds,
            "total": record.total_seconds,
        },
    }
    if candidate_policy_test_report is not None:
        event["candidate_policy_test_report"] = copy.deepcopy(
            dict(candidate_policy_test_report)
        )
    return event


def _official_records(
    observations: tuple[Mapping[str, Any], ...],
) -> list[dict[str, Any]]:
    """Match STATE-Bench's canonical {name, arguments, result} records."""

    return [
        {
            "name": observation["name"],
            "arguments": observation["arguments"],
            "result": observation["result"],
        }
        for observation in observations
    ]


def run_state_bench_task(
    *,
    task: Any,
    env_data: Any,
    user_id: str,
    domain: Any,
    agent_factory: Any,
    simulator: Any,
    mode: Mode,
    reviewer_factory: Any | None = None,
    verification_budget: int = 1,
    probability_aggregation: str = "vote_majority",
    probability_threshold: float = 0.5,
    probability_early_stop: bool = False,
    include_tool_schemas_in_evidence: bool = True,
    agent_episode_id: str,
    max_tool_rounds: int = 16,
    trajectory_metadata: Mapping[str, Any] | None = None,
    on_first_rejection: Callable[[], None] | None = None,
    on_review_event: Callable[[Mapping[str, Any]], None] | None = None,
    candidate_test_suite: PolicyTestSuite | None = None,
    first_rejection_diagnostic: Any | None = None,
) -> StateBenchRun:
    """Run one official task without exposing task or simulator oracles online.

    ``baseline`` and ``shadow`` share every conversation and environment code
    path.  The only branch is whether a write-capable batch is staged through
    ``ShadowVerifier`` or executed directly.

    Version one intentionally requires harness-executed agents.  A legacy
    ``StateBenchAgent.act()`` owns its tool execution and would bypass the
    publication barrier, so it fails explicitly.
    """

    if mode not in {"baseline", "shadow"}:
        raise ValueError("mode must be 'baseline' or 'shadow'")
    if type(include_tool_schemas_in_evidence) is not bool:
        raise TypeError("include_tool_schemas_in_evidence must be bool")
    if not callable(agent_factory):
        raise TypeError(
            "agent_factory must construct a BaseAgent from one sanitized runtime context"
        )
    if mode == "shadow" and reviewer_factory is None:
        raise ValueError("shadow mode requires reviewer_factory")
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be >= 1")
    if not callable(getattr(simulator, "respond", None)):
        raise TypeError("simulator must provide respond(public_conversation)")
    if not isinstance(agent_episode_id, str) or not agent_episode_id.strip():
        raise ValueError("agent_episode_id must be a non-empty opaque value")
    semantic_task_id = str(task.task_id)
    if semantic_task_id and semantic_task_id in agent_episode_id:
        raise ValueError("agent_episode_id must not contain the semantic task_id")
    if on_first_rejection is not None and not callable(on_first_rejection):
        raise TypeError("on_first_rejection must be callable or None")
    if on_review_event is not None and not callable(on_review_event):
        raise TypeError("on_review_event must be callable or None")
    if candidate_test_suite is not None and not isinstance(
        candidate_test_suite, PolicyTestSuite
    ):
        raise TypeError("candidate_test_suite must be PolicyTestSuite or None")
    if mode == "baseline" and candidate_test_suite is not None:
        raise ValueError("candidate policy tests are valid only in shadow mode")

    # Construct the agent only after the safe context exists.  Accepting an
    # already-created agent would be too late: its constructor could already
    # have copied official task oracles out of AgentRuntimeContext.
    safe_context = build_sanitized_runtime_context(
        episode_id=agent_episode_id,
        user_id=user_id,
        domain=domain.name,
        now=task.now,
    )
    agent = agent_factory(safe_context)
    if not isinstance(agent, BaseAgent):
        raise TypeError("agent_factory must return a state_bench BaseAgent")
    if agent.runtime_context is not safe_context:
        raise ValueError(
            "agent_factory must construct the agent with the supplied sanitized context"
        )
    if not agent.uses_harness_tool_execution():
        raise TypeError(
            "STATE-Bench Shadow workflow requires a harness-executed BaseAgent "
            "implementing generate_next_turn(); legacy self-executing agents can bypass the gate"
        )
    if agent.memory_tool_schemas() or agent.memory_tool_handlers():
        raise NotImplementedError(
            "v1 STATE-Bench workflow does not support agent-owned memory tools"
        )
    if type(agent).prepare_conversation is not BaseAgent.prepare_conversation:
        raise NotImplementedError(
            "STATE-Bench clean-context workflow forbids prepare_conversation overrides; "
            "shared retrieval needs an explicit evidence projection"
        )

    canonical_env = domain.environment_class(env_data.deep_copy(), now=task.now)
    initial_snapshot = canonical_env.get_full_snapshot()
    histories = StateBenchHistories()
    histories.append_shared({"role": "user", "content": task.opening_message})

    agent_system_prompt = domain.agent_system_prompt.format(
        now=task.now, user_id=user_id
    )
    # Producer receives disposable copies.  The Judge owns a separate frozen
    # episode copy so an agent cannot mutate shared schema objects and thereby
    # rewrite the policy surface seen by later reviews or sibling runs.
    judge_tool_schemas = tuple(copy.deepcopy(domain.tool_schemas))
    turn_judge_trace: list[dict[str, Any]] = []

    def context_provider() -> StateBenchEvidenceContext:
        # Earlier accepted calls in the same user turn are not yet compressed
        # into official history.  The Judge sees their native action/effect,
        # never the Producer's private intermediate narration.
        return StateBenchEvidenceContext(
            agent_system_prompt=agent_system_prompt,
            public_history=histories.public + tuple(turn_judge_trace),
            tool_schemas=(
                judge_tool_schemas if include_tool_schemas_in_evidence else None
            ),
        )

    candidate_provider = (
        StateBenchCandidateExecutionProvider(domain.name)
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
    executor = StateBenchForkExecutor(
        canonical=canonical_env,
        write_tool_names=domain.write_tool_names,
        context_provider=context_provider,
        candidate_execution_provider=candidate_provider,
        candidate_test_runner=candidate_runner,
    )
    shadow = (
        ShadowVerifier(
            executor,
            reviewer_factory,
            verification_budget=verification_budget,
            probability_aggregation=probability_aggregation,
            probability_threshold=probability_threshold,
            probability_early_stop=probability_early_stop,
        )
        if mode == "shadow"
        else None
    )

    accepted_tool_calls: list[dict[str, Any]] = []
    call_sequence = 0
    proposed_tool_calls = 0
    proposed_batches = 0
    rejected_batches = 0
    first_rejection_seen = False
    review_events: list[dict[str, Any]] = []

    for turn_index in range(domain.max_agent_turns):
        turn_judge_trace.clear()
        turn_public_calls: list[dict[str, Any]] = []
        # Match the official harness: prepare exactly once per user turn and
        # keep internal tool rounds only in this turn-local working context.
        working_conversation = agent.prepare_conversation(list(histories.producer))
        if not isinstance(working_conversation, list):
            raise TypeError("prepare_conversation() must return a list")
        final_text: str | None = None

        for tool_round in range(max_tool_rounds):
            response = agent.generate_next_turn(
                system_prompt=agent_system_prompt,
                conversation=copy.deepcopy(working_conversation),
                tools=copy.deepcopy(list(judge_tool_schemas)),
            )
            text, requested_calls = _normalize_response(response)
            if not requested_calls:
                final_text = text
                break

            native_calls: list[StateBenchToolCall] = []
            for index, requested in enumerate(requested_calls):
                call_sequence += 1
                native_calls.append(
                    StateBenchToolCall(
                        id=f"state:{turn_index + 1}:{tool_round + 1}:{index + 1}:{call_sequence}",
                        name=requested["name"],
                        arguments=requested["arguments"],
                    )
                )
            batch = StateBenchToolBatch(calls=tuple(native_calls))
            proposed_batches += 1
            proposed_tool_calls += len(native_calls)

            reviewed_this_batch = mode == "shadow" and executor.requires_review(batch) and not (
                first_rejection_diagnostic is not None and first_rejection_diagnostic.finished
            )
            if reviewed_this_batch:
                assert shadow is not None
                result = shadow.step(f"state-batch:{batch.fingerprint()}", batch)
                observations = result.observation
                published = isinstance(result, Published)
                record = shadow.audit_records[-1]
                if first_rejection_diagnostic is not None:
                    first_rejection_diagnostic.check_record(record)
                event = _review_event(
                    record=record,
                    batch=batch,
                    observations=observations,
                    producer_observation_enqueued=(
                        record.decision.code not in _REVIEW_INFRASTRUCTURE_CODES
                    ),
                    candidate_policy_test_report=(
                        executor.test_reports[-1]
                        if candidate_test_suite is not None
                        else None
                    ),
                )
                if record.decision.code in _REVIEW_INFRASTRUCTURE_CODES:
                    # The kernel has already discarded the fork. Do not turn a
                    # timeout/parser failure into semantic feedback, do not
                    # release paired replay, and do not let the episode proceed.
                    if on_review_event is not None:
                        on_review_event(copy.deepcopy(event))
                    raise StateBenchReviewInfrastructureError(
                        event,
                        prior_review_events=tuple(review_events),
                    )
                review_events.append(event)
            else:
                observations = executor.execute_direct(batch)
                published = True

            records = _official_records(observations)
            assistant_round = {
                "role": "assistant",
                "content": text,
                "tool_calls": records,
            }
            tool_round_message = {"role": "tool", "content": records}
            working_conversation.extend((assistant_round, tool_round_message))

            if published:
                # Judge-visible trace deliberately omits intermediate Producer
                # text, which could otherwise launder a rejected rationale into
                # the next fresh Judge's supposedly independent context.
                turn_judge_trace.extend(
                    (
                        {"role": "assistant", "tool_calls": records},
                        {"role": "tool", "content": records},
                    )
                )
                turn_public_calls.extend(records)
                accepted_tool_calls.extend(records)
            else:
                # This is the critical split: Producer receives repair feedback,
                # while UserSimulator, next fresh Judge, and official evaluator
                # never observe the rejected proposal.
                histories.append_rejected(assistant_round, tool_round_message)
                rejected_batches += 1
            if reviewed_this_batch:
                if on_review_event is not None:
                    on_review_event(copy.deepcopy(review_events[-1]))
            if not published:
                if not first_rejection_seen:
                    first_rejection_seen = True
                    if on_first_rejection is not None:
                        on_first_rejection()
        else:
            raise RuntimeError(f"Producer exceeded max_tool_rounds={max_tool_rounds}")

        if final_text is None:
            raise RuntimeError("Producer tool loop ended without a final response")

        canonical_assistant = {
            "role": "assistant",
            "content": final_text,
            "tool_calls": turn_public_calls or None,
        }
        # With no rejection, Producer and public histories are byte-equivalent
        # to the official compressed conversation.  Rejected overlay messages
        # already live only in Producer history and precede this final response.
        histories.append_shared(canonical_assistant)

        if turn_index >= domain.max_agent_turns - 1:
            break
        user_response = simulator.respond(list(histories.public))
        if not isinstance(user_response, str):
            raise TypeError("simulator.respond() must return text")
        histories.append_shared({"role": "user", "content": user_response})
        if domain.check_termination and domain.check_termination(user_response):
            break

    if first_rejection_diagnostic is not None:
        first_rejection_diagnostic.assert_complete()
    final_snapshot = executor.get_full_snapshot()
    state_diff = StateDiff.compute(initial_snapshot, final_snapshot)
    public_conversation = list(histories.public)
    efficiency = compute_efficiency(public_conversation, accepted_tool_calls)
    metadata = dict(trajectory_metadata or {})
    metadata.update({"workflow": "state_bench_shadow_v1", "execution_mode": mode})
    if first_rejection_diagnostic is not None:
        metadata["first_rejection_diagnostic"] = first_rejection_diagnostic.metadata
    token_usage = getattr(agent, "token_usage", None)
    if not isinstance(token_usage, TokenUsage):
        token_usage = TokenUsage()
    trajectory = Trajectory(
        task_id=task.task_id,
        user_id=user_id,
        task_summary=task.task_summary,
        conversation=public_conversation,
        state_diff=state_diff,
        efficiency=efficiency,
        token_usage=token_usage,
        metadata=metadata,
    )
    # Do not call agent.ingest_trajectory(): the official Trajectory carries
    # the semantic task ID and gold task summary.  Feeding it to cross-task
    # memory would cross the online oracle boundary.
    return StateBenchRun(
        mode=mode,
        trajectory=trajectory,
        producer_history=histories.producer,
        public_history=histories.public,
        audit_records=shadow.audit_records if shadow is not None else (),
        review_events=tuple(copy.deepcopy(review_events)),
        final_snapshot=final_snapshot,
        execution_metrics={
            "proposed_batches": proposed_batches,
            "proposed_tool_calls": proposed_tool_calls,
            "published_or_direct_tool_calls": len(accepted_tool_calls),
            "reviewed_batches": len(shadow.audit_records) if shadow is not None else 0,
            "review_votes": sum(
                len(record.review_decisions) for record in shadow.audit_records
            )
            if shadow is not None
            else 0,
            "rejected_batches": rejected_batches,
            "candidate_policy_test_reports": len(executor.test_reports),
            "candidate_policy_tests_failed": sum(
                report["summary"]["failed"] for report in executor.test_reports
            ),
            "candidate_policy_tests_error": sum(
                report["summary"]["error"] for report in executor.test_reports
            ),
            "judge_seconds": round(
                sum(record.review_seconds for record in shadow.audit_records), 6
            )
            if shadow is not None
            else 0.0,
        },
    )
