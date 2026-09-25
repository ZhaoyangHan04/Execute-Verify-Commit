"""Auditable τ² scoring with a frozen DashScope NL-assertion layer.

The upstream evaluator is used unchanged unless a task explicitly includes
``RewardType.NL_ASSERTION``.  For those tasks, the deterministic structural
components remain upstream τ² code while the upstream, implicitly configured
GPT-4.1 call is replaced by one explicitly configured and content-addressed
DashScope completion.  A first successful score is immutable: later scoring
of the same official task and semantic trajectory returns that exact result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from shadow_verifier.backends.dashscope import DashScopeBackend, DashScopeConfig
from shadow_verifier.experiments import (
    canonical_sha256,
    git_head,
    sha256_file,
)


SCORER_SCHEMA = "tau2-frozen-dashscope-nl-assertions-v1"
EVALUATION_SCHEMA = "tau2-official-structural-custom-nl-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
TAU2_ROOT = REPO_ROOT / "external" / "tau2-bench"

# Kept semantically equivalent to upstream
# tau2.evaluator.evaluator_nl_assertions.NLAssertionsEvaluator.  The final two
# constraints make response validation unambiguous without changing what is
# judged.
SYSTEM_PROMPT = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading

        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>
                }
            ]
        }

        - Return exactly one result per expected outcome, in input order.
        - Return JSON only and do not omit, duplicate, paraphrase, or reorder expected outcomes.
        """
USER_PROMPT_TEMPLATE = """
        conversation:
        {trajectory}

        expectedOutcomes:
        {assertions}
        """


@dataclass(frozen=True)
class FrozenNLEvaluatorConfig:
    """Every provider setting that can affect the frozen NL score."""

    model: str = "qwen3.8-max"
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 2048
    enable_thinking: bool = False
    seed: int = 42
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("NL evaluator model must be non-empty")
        if type(self.seed) is not int:
            raise TypeError("NL evaluator seed must be int")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("NL evaluator temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("NL evaluator top_p must be in (0, 1]")
        if self.max_completion_tokens <= 0:
            raise ValueError("NL evaluator max_completion_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("NL evaluator timeout_seconds must be positive")

    def backend_config(self) -> DashScopeConfig:
        return DashScopeConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
            enable_thinking=self.enable_thinking,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class FrozenEvaluationResult:
    """A reward plus scoring metadata that is deliberately outside it."""

    reward_info: Any
    audit: dict[str, Any]


def upstream_evaluator_provenance() -> dict[str, Any]:
    """Identify the exact upstream evaluator code whose components we reuse."""

    import tau2.evaluator.evaluator as evaluator_module
    import tau2.evaluator.evaluator_nl_assertions as nl_module

    files: dict[str, str] = {}
    for module in (evaluator_module, nl_module):
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(
                f"cannot locate upstream evaluator source: {module.__name__}"
            )
        path = Path(source).resolve()
        files[str(path.relative_to(TAU2_ROOT.resolve()))] = sha256_file(path)
    packages: dict[str, str | None] = {}
    for package in ("tau2-bench", "litellm", "openai"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    payload = {
        "tau2_git_head": git_head(TAU2_ROOT),
        "files": files,
        "packages": packages,
    }
    return {**payload, "fingerprint": canonical_sha256(payload)}


def scorer_provenance() -> dict[str, str]:
    payload = {
        "schema": SCORER_SCHEMA,
        "source_sha256": sha256_file(Path(__file__)),
        "system_prompt_sha256": canonical_sha256(SYSTEM_PROMPT),
        "user_prompt_template_sha256": canonical_sha256(USER_PROMPT_TEMPLATE),
    }
    return {**payload, "fingerprint": canonical_sha256(payload)}


def _json_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def canonical_trajectory_digest(simulation: Any) -> str:
    """Hash behavior while excluding timestamps, usage, and provider metadata."""

    projected: list[dict[str, Any]] = []
    for message in list(getattr(simulation, "messages", None) or ()):
        item: dict[str, Any] = {"role": _json_value(_field(message, "role"))}
        for name in ("content", "id", "requestor", "error"):
            value = _field(message, name)
            if value is not None:
                item[name] = _json_value(value)
        calls = _field(message, "tool_calls")
        if calls is not None:
            item["tool_calls"] = [
                {
                    "id": _field(call, "id"),
                    "name": _field(call, "name"),
                    "arguments": _field(call, "arguments"),
                    "requestor": _json_value(_field(call, "requestor")),
                }
                for call in calls
            ]
        nested = _field(message, "tool_messages")
        if nested is not None:
            item["tool_messages"] = [
                {
                    "id": _field(result, "id"),
                    "role": _json_value(_field(result, "role")),
                    "requestor": _json_value(_field(result, "requestor")),
                    "content": _field(result, "content"),
                    "error": _field(result, "error"),
                }
                for result in nested
            ]
        projected.append(item)
    termination = _json_value(getattr(simulation, "termination_reason", None))
    return canonical_sha256({"messages": projected, "termination_reason": termination})


def _strict_json_object(raw: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, ValueError):
        raise ValueError("NL evaluator returned invalid strict JSON") from None
    if not isinstance(value, dict):
        raise ValueError("NL evaluator response must be a JSON object")
    return value


def _parse_checks(raw: str, assertions: list[str]) -> list[Any]:
    from tau2.data_model.simulation import NLAssertionCheck

    payload = _strict_json_object(raw)
    if set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise ValueError("NL evaluator response must contain only a results list")
    rows = payload["results"]
    if len(rows) != len(assertions):
        raise ValueError("NL evaluator returned the wrong number of results")
    checks: list[Any] = []
    for index, (row, assertion) in enumerate(zip(rows, assertions, strict=True)):
        if not isinstance(row, dict) or set(row) != {
            "expectedOutcome",
            "reasoning",
            "metExpectation",
        }:
            raise ValueError(f"NL evaluator result {index} has an invalid schema")
        if row["expectedOutcome"] != assertion:
            raise ValueError(
                f"NL evaluator result {index} changed or reordered an assertion"
            )
        if type(row["metExpectation"]) is not bool:
            raise ValueError(f"NL evaluator result {index} metExpectation must be bool")
        if not isinstance(row["reasoning"], str) or not row["reasoning"].strip():
            raise ValueError(f"NL evaluator result {index} reasoning must be non-empty")
        checks.append(
            NLAssertionCheck(
                nl_assertion=assertion,
                met=row["metExpectation"],
                justification=row["reasoning"].strip(),
            )
        )
    return checks


def _prompt_messages(
    trajectory: list[Any], assertions: list[str]
) -> list[dict[str, str]]:
    trajectory_text = "\n".join(
        f"{_json_value(_field(message, 'role'))}: {_field(message, 'content')}"
        for message in trajectory
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                trajectory=trajectory_text,
                assertions=str(assertions),
            ),
        },
    ]


def _read_cache(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen NL evaluation cache is unreadable") from exc
    expected_key = canonical_sha256(identity)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCORER_SCHEMA
        or payload.get("cache_key") != expected_key
        or payload.get("identity") != identity
        or not isinstance(payload.get("nl_reward_info"), dict)
        or not isinstance(payload.get("completion"), dict)
    ):
        raise RuntimeError("frozen NL evaluation cache identity mismatch")
    return payload


def _publish_cache_once(path: Path, payload: dict[str, Any]) -> bool:
    """Publish a complete immutable JSON file without replacing a winner."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)


def _score_nl_assertions(
    *,
    simulation: Any,
    task: Any,
    task_fingerprint: str,
    domain: str,
    strict_replay: bool,
    cache_root: Path,
    config: FrozenNLEvaluatorConfig,
    backend_factory: Callable[[str, DashScopeConfig], Any] | None,
    upstream_provenance: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from tau2.data_model.simulation import RewardInfo
    from tau2.data_model.tasks import RewardType

    assertions = list(task.evaluation_criteria.nl_assertions or ())
    if not assertions:
        reward_info = RewardInfo(
            reward=1.0,
            nl_assertions=[],
            info={"note": "No nl_assertions to evaluate"},
            reward_breakdown={RewardType.NL_ASSERTION: 1.0},
        )
        return reward_info, {
            "cache_key": None,
            "cache_hit": False,
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_s": 0.0,
            "note": "no assertions",
        }

    trajectory = list(getattr(simulation, "messages", None) or ())
    messages = _prompt_messages(trajectory, assertions)
    scorer = scorer_provenance()
    identity = {
        "schema": SCORER_SCHEMA,
        "domain": domain,
        "task_fingerprint": task_fingerprint,
        "canonical_semantic_trajectory_sha256": canonical_trajectory_digest(simulation),
        "evaluation_type": "all",
        "communication_mode": "half_duplex",
        "solo_mode": False,
        "strict_replay": strict_replay,
        "model_config": asdict(config),
        "request_sha256": canonical_sha256(messages),
        "scorer_provenance": scorer,
        "upstream_evaluator_provenance": upstream_provenance,
    }
    cache_key = canonical_sha256(identity)
    cache_path = (
        cache_root
        / "_nl_assertion_cache"
        / SCORER_SCHEMA
        / cache_key[:2]
        / f"{cache_key}.json"
    )
    if cache_path.is_file():
        cached = _read_cache(cache_path, identity)
        return RewardInfo.model_validate(cached["nl_reward_info"]), {
            "cache_key": cache_key,
            "cache_hit": True,
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_s": 0.0,
            "frozen_completion": cached["completion"],
        }

    factory = backend_factory or (
        lambda model, backend_config: DashScopeBackend(
            model=model,
            config=backend_config,
        )
    )
    backend = factory(config.model, config.backend_config())
    completion = backend.complete(messages, seed=config.seed, json_mode=True)
    if not isinstance(completion.content, str):
        raise ValueError("NL evaluator returned no JSON content")
    checks = _parse_checks(completion.content, assertions)
    reward = 1.0 if all(check.met for check in checks) else 0.0
    reward_info = RewardInfo(
        reward=reward,
        nl_assertions=checks,
        reward_breakdown={RewardType.NL_ASSERTION: reward},
    )
    completion_record = {
        "response_sha256": hashlib.sha256(
            completion.content.encode("utf-8")
        ).hexdigest(),
        "model": completion.model,
        "finish_reason": completion.finish_reason,
        "usage": asdict(completion.usage),
        "latency_s": round(float(completion.latency_s), 6),
        "provenance": asdict(completion.provenance),
    }
    payload = {
        "schema": SCORER_SCHEMA,
        "cache_key": cache_key,
        "identity": identity,
        "nl_reward_info": reward_info.model_dump(mode="json"),
        "completion": completion_record,
    }
    won = _publish_cache_once(cache_path, payload)
    if not won:
        # A concurrent process established the immutable first result.  Return
        # that exact score, never the losing completion.
        cached = _read_cache(cache_path, identity)
        return RewardInfo.model_validate(cached["nl_reward_info"]), {
            "cache_key": cache_key,
            "cache_hit": True,
            "api_calls": 1,
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
            "latency_s": round(float(completion.latency_s), 6),
            "frozen_completion": cached["completion"],
            "concurrent_cache_loss": True,
        }
    return reward_info, {
        "cache_key": cache_key,
        "cache_hit": False,
        "api_calls": 1,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
        "latency_s": round(float(completion.latency_s), 6),
        "frozen_completion": completion_record,
    }


def _combine_reward_infos(
    task: Any,
    env_reward_info: Any,
    action_reward_info: Any,
    communicate_reward_info: Any,
    nl_reward_info: Any,
) -> Any:
    """Mirror the upstream ``EvaluationType.ALL`` combination exactly."""

    from tau2.data_model.simulation import RewardInfo
    from tau2.data_model.tasks import RewardType

    reward = 1.0
    env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
    action_bases = {RewardType.ACTION}
    nl_bases = {RewardType.NL_ASSERTION}
    comm_bases = {RewardType.COMMUNICATE}
    task_reward_basis = set(task.evaluation_criteria.reward_basis)
    evaluated_bases = env_bases | action_bases | nl_bases | comm_bases
    unevaluated = task_reward_basis - evaluated_bases
    if unevaluated:
        raise ValueError(f"Task reward_basis includes unevaluated types: {unevaluated}")

    reward_breakdown: dict[Any, float] = {}
    if task_reward_basis & env_bases:
        if env_reward_info.reward_breakdown is not None:
            reward_breakdown.update(env_reward_info.reward_breakdown)
        reward *= env_reward_info.reward
    if task_reward_basis & action_bases:
        if action_reward_info.reward_breakdown is not None:
            reward_breakdown.update(action_reward_info.reward_breakdown)
        reward *= action_reward_info.reward
    if task_reward_basis & nl_bases:
        if nl_reward_info.reward_breakdown is not None:
            reward_breakdown.update(nl_reward_info.reward_breakdown)
        reward *= nl_reward_info.reward
    if task_reward_basis & comm_bases:
        if communicate_reward_info.reward_breakdown is not None:
            reward_breakdown.update(communicate_reward_info.reward_breakdown)
        reward *= communicate_reward_info.reward

    return RewardInfo(
        reward=reward,
        db_check=env_reward_info.db_check,
        env_assertions=env_reward_info.env_assertions,
        action_checks=action_reward_info.action_checks,
        nl_assertions=nl_reward_info.nl_assertions,
        communicate_checks=communicate_reward_info.communicate_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=reward_breakdown,
        info={
            "env": env_reward_info.info,
            "nl": nl_reward_info.info,
            "communicate": communicate_reward_info.info,
            "action": action_reward_info.info,
        },
    )


def evaluate_tau2_simulation(
    *,
    simulation: Any,
    task: Any,
    task_fingerprint: str,
    domain: str,
    strict_replay: bool,
    cache_root: Path,
    config: FrozenNLEvaluatorConfig,
    backend_factory: Callable[[str, DashScopeConfig], Any] | None = None,
    official_evaluate: Callable[..., Any] | None = None,
    upstream_provenance: dict[str, Any] | None = None,
) -> FrozenEvaluationResult:
    """Evaluate one half-duplex run without an implicit evaluator model."""

    from tau2.data_model.simulation import TerminationReason
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode

    official = official_evaluate or evaluate_simulation
    criteria = task.evaluation_criteria
    task_needs_nl = bool(
        criteria is not None
        and RewardType.NL_ASSERTION in set(criteria.reward_basis or ())
    )
    evaluator_metadata = {
        "provider": "dashscope-openai-compatible",
        **asdict(config),
    }
    common_audit: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "score_protocol": (
            "tau2-official-structural+custom-dashscope-nl-v1"
            if task_needs_nl
            else "tau2-upstream-official-all"
        ),
        "task_requires_nl_assertion": task_needs_nl,
        "official_structural_evaluators": True,
        "custom_nl_evaluator": evaluator_metadata if task_needs_nl else None,
        "leaderboard_official": not task_needs_nl,
    }
    official_kwargs = {
        "simulation": simulation,
        "task": task,
        "solo_mode": False,
        "domain": domain,
        "mode": CommunicationMode.HALF_DUPLEX,
        "strict_replay": strict_replay,
    }
    valid_termination = simulation.termination_reason in {
        TerminationReason.AGENT_STOP,
        TerminationReason.USER_STOP,
    }
    if not task_needs_nl or not valid_termination or criteria is None:
        reward_info = official(
            evaluation_type=EvaluationType.ALL,
            **official_kwargs,
        )
        reason = (
            "not_required"
            if not task_needs_nl
            else (
                "premature_termination"
                if not valid_termination
                else "no_evaluation_criteria"
            )
        )
        return FrozenEvaluationResult(
            reward_info=reward_info,
            audit={
                **common_audit,
                "upstream_all_used": True,
                "nl_evaluation": {
                    "status": "skipped",
                    "reason": reason,
                    "api_calls": 0,
                    "cache_hit": False,
                },
            },
        )

    env_reward = official(evaluation_type=EvaluationType.ENV, **official_kwargs)
    action_reward = official(evaluation_type=EvaluationType.ACTION, **official_kwargs)
    communicate_reward = official(
        evaluation_type=EvaluationType.COMMUNICATE,
        **official_kwargs,
    )
    provenance = upstream_provenance or upstream_evaluator_provenance()
    nl_reward, nl_audit = _score_nl_assertions(
        simulation=simulation,
        task=task,
        task_fingerprint=task_fingerprint,
        domain=domain,
        strict_replay=strict_replay,
        cache_root=cache_root,
        config=config,
        backend_factory=backend_factory,
        upstream_provenance=provenance,
    )
    reward_info = _combine_reward_infos(
        task,
        env_reward,
        action_reward,
        communicate_reward,
        nl_reward,
    )
    return FrozenEvaluationResult(
        reward_info=reward_info,
        audit={
            **common_audit,
            "upstream_all_used": False,
            "upstream_evaluator_provenance": provenance,
            "nl_evaluation": {"status": "complete", **nl_audit},
        },
    )
