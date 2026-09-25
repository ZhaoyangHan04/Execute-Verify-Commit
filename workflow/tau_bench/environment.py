"""Fork, review, and promote one original τ-bench mutation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from shadow_verifier import Candidate, Decision

from .evidence import build_evidence
from .model import TauBenchToolAction


@dataclass
class _CandidateHandle:
    owner: object
    data: dict[str, Any] | None
    observation: str
    before_hash: str
    before_generation: int
    published: bool = False
    discarded: bool = False


class TauBenchForkExecutor:
    """Stage native tools against a deep-copied DB and promote that exact copy."""

    def __init__(
        self,
        environment: Any,
        *,
        policy: str,
        context_provider: Callable[[], Sequence[dict[str, Any]]],
        tool_schemas: Sequence[Mapping[str, Any]],
        include_tool_schemas_in_evidence: bool = True,
        policy_presentation: str = "full",
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.context_provider = context_provider
        if any(not isinstance(schema, Mapping) for schema in tool_schemas):
            raise TypeError("tool_schemas must contain mappings")
        if type(include_tool_schemas_in_evidence) is not bool:
            raise TypeError("include_tool_schemas_in_evidence must be bool")
        # Freeze the exact producer-visible tool surface for the episode.  The
        # reviewer gets disposable copies through Evidence and cannot rewrite
        # schemas used by the producer or a later candidate.
        self.tool_schemas = tuple(copy.deepcopy(schema) for schema in tool_schemas)
        self.include_tool_schemas_in_evidence = include_tool_schemas_in_evidence
        if policy_presentation not in {"full", "action_focus_v1"}:
            raise ValueError("unknown policy presentation")
        self.policy_presentation = policy_presentation
        self._owner = object()
        self._generation = 0
        self._sequence = 0
        self._active: dict[str, _CandidateHandle] = {}

    def fingerprint(self, action: TauBenchToolAction, /) -> str:
        if not isinstance(action, TauBenchToolAction):
            raise TypeError("action must be TauBenchToolAction")
        return action.fingerprint()

    def stage(
        self,
        call_id: str,
        action: TauBenchToolAction,
        /,
    ) -> Candidate[_CandidateHandle]:
        if action.name not in self.environment.tools_map:
            raise ValueError(f"unknown τ-bench tool: {action.name}")
        before_state = copy.deepcopy(self.environment.data)
        candidate_data = copy.deepcopy(self.environment.data)
        try:
            observation = self.environment.tools_map[action.name].invoke(
                data=candidate_data,
                **action.arguments,
            )
        except Exception as exc:
            observation = f"Error: {exc}"
        if not isinstance(observation, str):
            observation = str(observation)
        before_hash = self._data_hash(self.environment.data)
        self._sequence += 1
        candidate_id = hashlib.sha256(
            (
                f"{call_id}\0{self._sequence}\0{action.fingerprint()}\0{before_hash}"
            ).encode("utf-8")
        ).hexdigest()
        handle = _CandidateHandle(
            owner=self._owner,
            data=candidate_data,
            observation=observation,
            before_hash=before_hash,
            before_generation=self._generation,
        )
        self._active[candidate_id] = handle
        return Candidate(
            call_id=call_id,
            candidate_id=candidate_id,
            handle=handle,
            evidence=build_evidence(
                policy=self.policy,
                public_history=self.context_provider(),
                tool_schemas=(
                    self.tool_schemas
                    if self.include_tool_schemas_in_evidence
                    else None
                ),
                action=action,
                observation=observation,
                before_state=before_state,
                after_state=candidate_data,
                policy_presentation=self.policy_presentation,
            ),
        )

    def publish(self, candidate: Candidate[_CandidateHandle], /) -> str:
        handle = self._require_active(candidate)
        if handle.data is None:
            raise RuntimeError("candidate data is unavailable")
        if (
            self._generation != handle.before_generation
            or self._data_hash(self.environment.data) != handle.before_hash
        ):
            raise RuntimeError("canonical τ-bench state changed before publish")
        self.environment.data = handle.data
        self._generation += 1
        handle.published = True
        self._active.pop(candidate.candidate_id, None)
        return handle.observation

    def rejection(
        self,
        candidate: Candidate[_CandidateHandle],
        decision: Decision,
        /,
    ) -> str:
        handle = self._require_active(candidate)
        if decision.accept:
            raise ValueError("rejection requires a rejected decision")
        return json.dumps(
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

    def discard(self, candidate: Candidate[_CandidateHandle], /) -> None:
        handle = self._require_active(candidate)
        handle.data = None
        handle.discarded = True
        self._active.pop(candidate.candidate_id, None)

    def _require_active(
        self,
        candidate: Candidate[_CandidateHandle],
    ) -> _CandidateHandle:
        if not isinstance(candidate, Candidate):
            raise TypeError("candidate must be Candidate")
        handle = candidate.handle
        if not isinstance(handle, _CandidateHandle) or handle.owner is not self._owner:
            raise ValueError("candidate belongs to another executor")
        if self._active.get(candidate.candidate_id) is not handle:
            raise RuntimeError("candidate is not active")
        return handle

    @staticmethod
    def _data_hash(data: dict[str, Any]) -> str:
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
