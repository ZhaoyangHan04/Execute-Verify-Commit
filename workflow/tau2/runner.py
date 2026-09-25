"""Reproducible baseline-vs-Shadow pilot on official τ² tasks.

Online execution never passes a ``Task`` to the Judge adapter.  Ground-truth
criteria are used only after an episode has terminated, by the pinned official
τ² evaluator.  Shadow runs retain two trajectories:

* producer: includes rejected proposals and bounded decision feedback;
* canonical: removes those proposal/result batches for strict official replay.
"""

from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
from shadow_verifier.backends.dashscope import DashScopeBackend, DashScopeConfig
from shadow_verifier.backends.replay import ReplayCacheBackend
from shadow_verifier.backends.hosted import hosted_backend
from shadow_verifier.policy_tests import PolicyTestSuite
from shadow_verifier.experiments import (
    atomic_json_dump,
    canonical_sha256 as _canonical_sha256,
    git_head as _git_head,
    require_clean_git_tree as _require_clean_git_tree,
    sha256_file as _sha256_file,
)
from shadow_verifier.reviewers import (
    MAX_REVIEW_COMPLETION_TOKENS,
    REVIEWER_PROMPT_VARIANTS,
    SemanticReviewerConfig,
    SemanticReviewerFactory,
    reviewer_prompt,
)
from shadow_verifier import MAX_VERIFICATION_BUDGET, PROBABILITY_AGGREGATIONS

Mode = Literal["baseline", "shadow"]
PairingMode = Literal["independent", "record", "replay"]
Domain = Literal["airline", "retail", "telecom"]
PILOT_TASK_IDS: tuple[str, ...] = tuple((str(index) for index in range(10)))
DOMAIN: Domain = "airline"
SUPPORTED_DOMAINS: tuple[Domain, ...] = ("airline", "retail", "telecom")
REPO_ROOT = Path(__file__).resolve().parents[2]
TAU2_ROOT = REPO_ROOT / "external" / "tau2-bench"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "tau2"
RUNNER_SCHEMA = "shadow-tau2-multidomain-v7-reviewer-variants"
_PAIR_ID_PATTERN = re.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class ResultArtifactConflictError(RuntimeError):
    """An existing result belongs to a different experiment identity."""


def _task_slug(domain: Domain, task_id: str) -> str:
    """Return a short filesystem-safe identity without embedding raw task IDs."""
    digest = hashlib.sha256(f"{domain}\x00{task_id}".encode("utf-8")).hexdigest()[:24]
    return f"{domain}-{digest}"


@dataclass(frozen=True)
class Tau2PilotConfig:
    """Frozen behavior-affecting settings for a paired pilot cell."""

    producer_model: str
    judge_model: str
    mode: Mode
    reviewer_variant: str = "plain_checks_v1"
    domain: Domain = DOMAIN
    task_split: str = "base"
    sample_manifest_fingerprint: str | None = None
    user_model: str = "qwen-max"
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 8192
    judge_max_completion_tokens: int = MAX_REVIEW_COMPLETION_TOKENS
    verification_budget: int = 1
    reviewer_temperature: float = 0.0
    reviewer_seed_mode: str = "fixed"
    reviewer_soft_label_output: bool = False
    reviewer_probability_aggregation: str = "vote_majority"
    reviewer_probability_threshold: float = 0.5
    reviewer_probability_early_stop: bool = False
    reviewer_excluded_from_pair_common: bool = False
    enable_thinking: bool = False
    nl_evaluator_model: str = "qwen3.8-max"
    nl_evaluator_temperature: float = 0.0
    nl_evaluator_top_p: float = 1.0
    nl_evaluator_max_completion_tokens: int = 2048
    nl_evaluator_enable_thinking: bool = False
    nl_evaluator_seed: int = 42
    max_steps: int = 200
    max_errors: int = 10
    episode_timeout_seconds: float = 1800.0
    request_timeout_seconds: float = 300.0
    validate_communication: bool = True
    require_paired_cache_hits: bool = False
    paired_cache_source_pair_run_id: str | None = None
    paired_cache_source_manifest_fingerprint: str | None = None
    pairing_mode: PairingMode = "independent"
    pair_run_id: str | None = None
    paired_task_ids: tuple[str, ...] | None = None
    candidate_test_suite_sha256: str | None = None
    candidate_test_compiler_model: str | None = None
    include_tool_schemas_in_evidence: bool = True
    reviewer_candidate_effect_visibility: str = "full"

    def __post_init__(self) -> None:
        for field_name in (
            "producer_model",
            "judge_model",
            "user_model",
            "nl_evaluator_model",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.mode not in ("baseline", "shadow"):
            raise ValueError("mode must be baseline or shadow")
        if self.reviewer_variant not in REVIEWER_PROMPT_VARIANTS:
            raise ValueError(
                "reviewer_variant must be one of " + ", ".join(REVIEWER_PROMPT_VARIANTS)
            )
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError(f"domain must be one of {', '.join(SUPPORTED_DOMAINS)}")
        if not isinstance(self.task_split, str) or not self.task_split.strip():
            raise ValueError("task_split must be non-empty")
        if (
            self.sample_manifest_fingerprint is not None
            and re.fullmatch("[0-9a-f]{64}", self.sample_manifest_fingerprint) is None
        ):
            raise ValueError("sample_manifest_fingerprint must be lowercase SHA-256")
        if self.pairing_mode not in ("independent", "record", "replay"):
            raise ValueError("invalid pairing_mode")
        if self.pairing_mode == "independent":
            if self.pair_run_id is not None or self.paired_task_ids is not None:
                raise ValueError(
                    "independent runs cannot set pair_run_id or paired_task_ids"
                )
        else:
            if (
                not isinstance(self.pair_run_id, str)
                or _PAIR_ID_PATTERN.fullmatch(self.pair_run_id) is None
            ):
                raise ValueError("paired runs require a safe non-empty pair_run_id")
            if (
                not isinstance(self.paired_task_ids, tuple)
                or not self.paired_task_ids
                or (
                    not all(
                        (
                            isinstance(task_id, str) and task_id
                            for task_id in self.paired_task_ids
                        )
                    )
                )
                or (len(set(self.paired_task_ids)) != len(self.paired_task_ids))
            ):
                raise ValueError(
                    "paired runs require a non-empty unique ordered task tuple"
                )
            if self.pairing_mode == "record" and self.mode != "baseline":
                raise ValueError("record pairing_mode is baseline-only")
            if self.pairing_mode == "replay" and self.mode != "shadow":
                raise ValueError("replay pairing_mode is shadow-only")
        if type(self.seed) is not int:
            raise TypeError("seed must be int")
        if type(self.include_tool_schemas_in_evidence) is not bool:
            raise TypeError("include_tool_schemas_in_evidence must be bool")
        if self.reviewer_candidate_effect_visibility not in {"full", "hidden"}:
            raise ValueError(
                "reviewer_candidate_effect_visibility must be full or hidden"
            )
        if type(self.reviewer_excluded_from_pair_common) is not bool:
            raise TypeError("reviewer_excluded_from_pair_common must be bool")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.judge_max_completion_tokens <= 0:
            raise ValueError("judge_max_completion_tokens must be positive")
        if self.judge_max_completion_tokens > MAX_REVIEW_COMPLETION_TOKENS:
            raise ValueError(
                "judge_max_completion_tokens exceeds the reviewer response budget"
            )
        if type(self.verification_budget) is not int:
            raise TypeError("verification_budget must be int")
        if (
            self.verification_budget <= 0
            or self.verification_budget > MAX_VERIFICATION_BUDGET
            or self.verification_budget % 2 == 0
        ):
            raise ValueError(
                f"verification_budget must be an odd integer in [1, {MAX_VERIFICATION_BUDGET}]"
            )
        if self.mode == "baseline" and self.verification_budget != 1:
            raise ValueError("baseline mode requires verification_budget=1")
        if not 0.0 <= self.reviewer_temperature <= 2.0:
            raise ValueError("reviewer_temperature must be in [0, 2]")
        if self.reviewer_seed_mode not in {"independent", "fixed"}:
            raise ValueError("reviewer_seed_mode must be 'independent' or 'fixed'")
        if type(self.reviewer_soft_label_output) is not bool:
            raise TypeError("reviewer_soft_label_output must be bool")
        if type(self.reviewer_probability_early_stop) is not bool:
            raise TypeError("reviewer_probability_early_stop must be bool")
        if self.reviewer_probability_aggregation not in PROBABILITY_AGGREGATIONS:
            raise ValueError("unknown reviewer_probability_aggregation")
        if not 0.5 <= self.reviewer_probability_threshold <= 1.0:
            raise ValueError("reviewer_probability_threshold must be in [0.5, 1]")
        if (
            self.reviewer_probability_aggregation != "mean_probability"
            and self.reviewer_probability_threshold != 0.5
        ):
            raise ValueError("non-mean probability aggregation requires threshold 0.5")
        if self.reviewer_probability_aggregation != "vote_majority" and (
            not self.reviewer_soft_label_output
        ):
            raise ValueError(
                "probability aggregation requires reviewer_soft_label_output"
            )
        if self.reviewer_probability_early_stop and (
            not self.reviewer_soft_label_output
        ):
            raise ValueError(
                "reviewer_probability_early_stop requires reviewer_soft_label_output"
            )
        if (
            self.reviewer_probability_early_stop
            and self.reviewer_probability_aggregation != "mean_probability"
        ):
            raise ValueError(
                "reviewer_probability_early_stop requires mean_probability aggregation"
            )
        if self.reviewer_probability_early_stop and self.verification_budget == 1:
            raise ValueError(
                "reviewer_probability_early_stop requires verification_budget > 1"
            )
        if type(self.nl_evaluator_seed) is not int:
            raise TypeError("nl_evaluator_seed must be int")
        if not 0.0 <= self.nl_evaluator_temperature <= 2.0:
            raise ValueError("nl_evaluator_temperature must be in [0, 2]")
        if not 0.0 < self.nl_evaluator_top_p <= 1.0:
            raise ValueError("nl_evaluator_top_p must be in (0, 1]")
        if self.nl_evaluator_max_completion_tokens <= 0:
            raise ValueError("nl_evaluator_max_completion_tokens must be positive")
        if self.max_steps <= 0 or self.max_errors <= 0:
            raise ValueError("max_steps and max_errors must be positive")
        if type(self.require_paired_cache_hits) is not bool:
            raise TypeError("require_paired_cache_hits must be bool")
        if self.pairing_mode == "independent" and self.require_paired_cache_hits:
            raise ValueError("require_paired_cache_hits is valid only for paired runs")
        cache_source = (
            self.paired_cache_source_pair_run_id,
            self.paired_cache_source_manifest_fingerprint,
        )
        if (cache_source[0] is None) != (cache_source[1] is None):
            raise ValueError("paired cache source identity must be all-or-none")
        if cache_source[0] is not None:
            if self.pairing_mode == "independent":
                raise ValueError("paired cache source is valid only for paired runs")
            if not self.require_paired_cache_hits:
                raise ValueError("paired cache source requires fail-closed cache hits")
            if (
                type(cache_source[0]) is not str
                or _PAIR_ID_PATTERN.fullmatch(cache_source[0]) is None
            ):
                raise ValueError("paired cache source pair_run_id is invalid")
            if (
                type(cache_source[1]) is not str
                or re.fullmatch("[0-9a-f]{64}", cache_source[1]) is None
            ):
                raise ValueError("paired cache source manifest fingerprint is invalid")
        if self.episode_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        suite_fields = (
            self.candidate_test_suite_sha256,
            self.candidate_test_compiler_model,
        )
        if (suite_fields[0] is None) != (suite_fields[1] is None):
            raise ValueError("candidate policy-test suite identity must be all-or-none")
        if suite_fields[0] is not None:
            if re.fullmatch("[0-9a-f]{64}", suite_fields[0]) is None:
                raise ValueError(
                    "candidate_test_suite_sha256 must be lowercase SHA-256"
                )
            if suite_fields[1] != self.producer_model:
                raise ValueError(
                    "candidate policy-test compiler model must equal producer_model"
                )

    @property
    def artifact_config(self) -> dict[str, Any]:
        value = asdict(self)
        if not self.reviewer_soft_label_output:
            value.pop("reviewer_soft_label_output")
        if self.reviewer_probability_aggregation == "vote_majority":
            value.pop("reviewer_probability_aggregation")
        if self.reviewer_probability_threshold == 0.5:
            value.pop("reviewer_probability_threshold")
        if not self.reviewer_probability_early_stop:
            value.pop("reviewer_probability_early_stop")
        if not self.reviewer_excluded_from_pair_common:
            value.pop("reviewer_excluded_from_pair_common")
        if self.reviewer_candidate_effect_visibility == "full":
            value.pop("reviewer_candidate_effect_visibility")
        return value

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(
            {"schema": RUNNER_SCHEMA, "config": self.artifact_config}
        )

    @property
    def pair_common_fingerprint(self) -> str | None:
        if self.pairing_mode == "independent":
            return None
        shared = self.artifact_config
        shared.pop("mode")
        shared.pop("pairing_mode")
        if self.reviewer_excluded_from_pair_common:
            shared.pop("judge_model")
        shared.pop("reviewer_variant")
        shared.pop("verification_budget")
        shared.pop("reviewer_temperature")
        shared.pop("reviewer_seed_mode")
        shared.pop("reviewer_soft_label_output", None)
        shared.pop("reviewer_probability_aggregation", None)
        shared.pop("reviewer_probability_threshold", None)
        shared.pop("reviewer_probability_early_stop", None)
        shared.pop("include_tool_schemas_in_evidence")
        shared.pop("reviewer_candidate_effect_visibility", None)
        return _canonical_sha256({"schema": RUNNER_SCHEMA, "paired_config": shared})

    @property
    def reviewer_system_prompt_sha256(self) -> str:
        return hashlib.sha256(
            reviewer_prompt(self.reviewer_variant).encode("utf-8")
        ).hexdigest()


def data_provenance(
    domain: Domain = DOMAIN, task_split: str = "base"
) -> dict[str, Any]:
    _require_clean_git_tree(TAU2_ROOT, subject="pinned upstream tau2 worktree")
    spec = importlib.util.find_spec("tau2")
    imported_from: str | None = None
    if spec is not None and spec.origin is not None:
        imported_path = Path(spec.origin).resolve()
        try:
            imported_path.relative_to(TAU2_ROOT.resolve())
        except ValueError:
            raise RuntimeError("imported tau2 package is not the pinned checkout")
        imported_from = str(imported_path.relative_to(REPO_ROOT))
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported tau2 domain: {domain!r}")
    domain_root = TAU2_ROOT / "data" / "tau2" / "domains" / domain
    domain_files = {
        "airline": ("tasks.json", "db.json", "policy.md", "split_tasks.json"),
        "retail": ("tasks.json", "db.json", "policy.md", "split_tasks.json"),
        "telecom": (
            "tasks.json",
            "db.toml",
            "user_db.toml",
            "main_policy.md",
            "tech_support_manual.md",
            "split_tasks.json",
        ),
    }
    behavior_files = [
        *(domain_root / name for name in domain_files[domain]),
        *sorted((TAU2_ROOT / "data" / "tau2" / "user_simulator").glob("*.md")),
        TAU2_ROOT / "pyproject.toml",
        TAU2_ROOT / "uv.lock",
    ]
    return {
        "domain": domain,
        "task_split": task_split,
        "tau2_git_head": _git_head(TAU2_ROOT),
        "tau2_git_clean": True,
        "tau2_imported_from": imported_from,
        "files": {
            str(path.relative_to(TAU2_ROOT)): _sha256_file(path)
            for path in behavior_files
        },
    }


def implementation_provenance() -> dict[str, Any]:
    """Hash every behavior-bearing source file used by this experiment."""
    roots = (REPO_ROOT / "shadow-verifier" / "src", REPO_ROOT / "workflow" / "tau2")
    files: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            local_parts = path.relative_to(root).parts
            if "tests" in local_parts or "cli" in local_parts:
                continue
            relative = str(path.relative_to(REPO_ROOT))
            files[relative] = _sha256_file(path)
    package_manifest = REPO_ROOT / "pyproject.toml"
    files[str(package_manifest.relative_to(REPO_ROOT))] = _sha256_file(package_manifest)
    try:
        openai_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        openai_version = None
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "openai_version": openai_version,
    }
    return {
        "files": files,
        "runtime": runtime,
        "fingerprint": _canonical_sha256({"files": files, "runtime": runtime}),
    }


def ensure_pair_manifest(
    output_root: Path, baseline_config: Tau2PilotConfig, shadow_config: Tau2PilotConfig
) -> dict[str, Any]:
    """Create or verify the immutable identity of one paired experiment."""
    if (
        baseline_config.pairing_mode != "record"
        or shadow_config.pairing_mode != "replay"
        or baseline_config.pair_run_id != shadow_config.pair_run_id
        or (baseline_config.paired_task_ids != shadow_config.paired_task_ids)
        or (
            baseline_config.pair_common_fingerprint
            != shadow_config.pair_common_fingerprint
        )
    ):
        raise ValueError("configs do not define one valid paired experiment")
    if baseline_config.pair_run_id is None:
        raise ValueError("pair manifest requires pair_run_id")
    data = data_provenance(baseline_config.domain, baseline_config.task_split)
    implementation = implementation_provenance()
    payload = {
        "schema": "shadow-tau2-pair-manifest-v3-reviewer-variant",
        "pair_run_id": baseline_config.pair_run_id,
        "domain": baseline_config.domain,
        "task_split": baseline_config.task_split,
        "sample_manifest_fingerprint": baseline_config.sample_manifest_fingerprint,
        "ordered_task_ids": list(baseline_config.paired_task_ids or ()),
        "pair_common_fingerprint": baseline_config.pair_common_fingerprint,
        "baseline_reviewer_variant": baseline_config.reviewer_variant,
        "shadow_reviewer_variant": shadow_config.reviewer_variant,
        "baseline_verification_budget": baseline_config.verification_budget,
        "shadow_verification_budget": shadow_config.verification_budget,
        **{},
        "shadow_reviewer_system_prompt_sha256": shadow_config.reviewer_system_prompt_sha256,
        "completion_cache_source": {
            "pair_run_id": baseline_config.paired_cache_source_pair_run_id,
            "manifest_fingerprint": baseline_config.paired_cache_source_manifest_fingerprint,
        }
        if baseline_config.paired_cache_source_pair_run_id is not None
        else None,
        "baseline_config_fingerprint": baseline_config.fingerprint,
        "shadow_config_fingerprint": shadow_config.fingerprint,
        "data_provenance_fingerprint": _canonical_sha256(data),
        "implementation_fingerprint": implementation["fingerprint"],
    }
    payload["manifest_fingerprint"] = _canonical_sha256(payload)
    path = (
        output_root
        / "_paired_completion_cache"
        / baseline_config.domain
        / baseline_config.pair_run_id
        / "manifest.json"
    )
    path.parent.mkdir(mode=448, parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("existing pair manifest is unreadable") from exc
        if existing != payload:
            raise RuntimeError("pair_run_id is already bound to a different experiment")
        return payload
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return payload


def load_pilot_tasks(
    task_ids: Sequence[str] = PILOT_TASK_IDS,
    *,
    domain: Domain = DOMAIN,
    task_split: str = "base",
) -> list[Any]:
    from tau2.runner.helpers import get_tasks

    requested = tuple((str(task_id) for task_id in task_ids))
    if not requested:
        raise ValueError("at least one task ID is required")
    if len(set(requested)) != len(requested):
        raise ValueError("task IDs must be unique")
    tasks = get_tasks(domain, task_split_name=task_split, task_ids=list(requested))
    by_id = {task.id: task for task in tasks}
    return [by_id[task_id] for task_id in requested]


def _task_fingerprint(task: Any) -> str:
    dumper = getattr(task, "model_dump", None)
    if not callable(dumper):
        raise TypeError("tau2 task must provide model_dump()")
    return _canonical_sha256(dumper(mode="json"))


def _user_tools(environment: Any, task: Any) -> list[Any] | None:
    if environment.user_tools is None:
        return None
    return environment.get_user_tools(include=task.user_tools) or None


def _backend(
    model: str,
    config: Tau2PilotConfig,
    *,
    output_root: Path,
    task_id: str,
    participant_role: Literal["producer", "user", "judge"],
    judge: bool = False,
):
    backend = hosted_backend(
        model=model,
        default_factory=DashScopeBackend,
        role="verifier" if judge else participant_role,
        config=DashScopeConfig(
            temperature=config.reviewer_temperature if judge else config.temperature,
            top_p=1.0 if judge else config.top_p,
            max_completion_tokens=config.judge_max_completion_tokens
            if judge
            else config.max_completion_tokens,
            enable_thinking=False if judge else config.enable_thinking,
            timeout_seconds=config.request_timeout_seconds,
        ),
    )
    if judge or config.pairing_mode == "independent":
        return backend
    if config.pair_run_id is None:
        raise RuntimeError("paired backend is missing pair_run_id")
    cache_namespace = paired_cache_namespace(config)
    return ReplayCacheBackend(
        backend,
        cache_dir=output_root
        / "_paired_completion_cache"
        / config.domain
        / config.pair_run_id
        / cache_namespace
        / _task_slug(config.domain, task_id)
        / participant_role,
        replay_only=config.pairing_mode == "replay"
        or (config.pairing_mode == "record" and config.require_paired_cache_hits),
    )


def paired_cache_namespace(config: Tau2PilotConfig) -> str:
    """Return the strict Producer/User completion-cache namespace."""
    if config.pairing_mode == "independent":
        raise ValueError("independent runs do not have a paired cache namespace")
    return _canonical_sha256(
        {
            "pair_common_fingerprint": config.pair_common_fingerprint,
            "data": _canonical_sha256(
                data_provenance(config.domain, config.task_split)
            ),
            "implementation": implementation_provenance()["fingerprint"],
        }
    )


def _normalise_turn_indices(messages: Sequence[Any]) -> list[Any]:
    result = copy.deepcopy(list(messages))
    for index, message in enumerate(result):
        message.turn_idx = index
    return result


def _evaluate(
    simulation: Any,
    task: Any,
    *,
    config: Tau2PilotConfig,
    output_root: Path,
    task_fingerprint: str,
    strict_replay: bool,
) -> Any:
    from .evaluation import FrozenNLEvaluatorConfig, evaluate_tau2_simulation

    return evaluate_tau2_simulation(
        simulation=simulation,
        task=task,
        task_fingerprint=task_fingerprint,
        domain=config.domain,
        strict_replay=strict_replay,
        cache_root=output_root,
        config=FrozenNLEvaluatorConfig(
            model=config.nl_evaluator_model,
            temperature=config.nl_evaluator_temperature,
            top_p=config.nl_evaluator_top_p,
            max_completion_tokens=config.nl_evaluator_max_completion_tokens,
            enable_thinking=config.nl_evaluator_enable_thinking,
            seed=config.nl_evaluator_seed,
            timeout_seconds=config.request_timeout_seconds,
        ),
    )


def _message_metrics(messages: Sequence[Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "agent_model_invocations": 0,
        "user_model_invocations": 0,
        "agent_api_calls": 0,
        "user_api_calls": 0,
        "agent_replay_cache_hits": 0,
        "user_replay_cache_hits": 0,
        "agent_prompt_tokens": 0,
        "agent_completion_tokens": 0,
        "agent_total_tokens": 0,
        "user_prompt_tokens": 0,
        "user_completion_tokens": 0,
        "user_total_tokens": 0,
        "agent_generation_seconds": 0.0,
        "user_generation_seconds": 0.0,
    }
    for message in messages:
        role = getattr(message, "role", None)
        if role not in ("assistant", "user"):
            continue
        usage = getattr(message, "usage", None)
        if not isinstance(usage, dict):
            continue
        prefix = "agent" if role == "assistant" else "user"
        metrics[f"{prefix}_model_invocations"] += 1
        raw_data = getattr(message, "raw_data", None)
        cache_hit = bool(
            isinstance(raw_data, dict) and raw_data.get("replay_cache_hit") is True
        )
        if cache_hit:
            metrics[f"{prefix}_replay_cache_hits"] += 1
        else:
            metrics[f"{prefix}_api_calls"] += 1
        for token_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            metrics[f"{prefix}_{token_name}"] += int(usage.get(token_name, 0) or 0)
        metrics[f"{prefix}_generation_seconds"] += float(
            getattr(message, "generation_time_seconds", 0.0) or 0.0
        )
    for key in ("agent_generation_seconds", "user_generation_seconds"):
        metrics[key] = round(metrics[key], 6)
    return metrics


def _mutating_calls(messages: Sequence[Any], environment: Any) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        calls = tuple(getattr(message, "tool_calls", None) or ())
        mutating = [call for call in calls if environment._is_mutating_tool(call.name)]
        if not mutating:
            continue
        batches.append(
            {
                "tool_call_ids": [call.id for call in calls],
                "mutating_tool_call_ids": [call.id for call in mutating],
                "mutating_tool_names": [call.name for call in mutating],
            }
        )
    return batches


def result_path(output_root: Path, config: Tau2PilotConfig, task_id: str) -> Path:
    model_component = config.producer_model.replace("/", "_")
    judge_component = config.judge_model.replace("/", "_")
    directory = output_root / "tau2" / config.domain / model_component / config.mode
    if config.mode == "shadow":
        directory /= f"variant-{config.reviewer_variant}"
        if config.verification_budget > 1:
            directory /= f"verification-budget-{config.verification_budget}"
        if config.reviewer_candidate_effect_visibility != "full":
            directory /= "candidate-effect-hidden"
    directory = directory / f"judge-{judge_component}" / f"s{config.seed}"
    if config.pair_run_id is not None:
        directory /= f"pair-{config.pair_run_id}"
    return directory / _task_slug(config.domain, task_id) / "result.json"


def run_task(
    *,
    task: Any,
    config: Tau2PilotConfig,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    resume: bool = True,
    candidate_test_suite: PolicyTestSuite | None = None,
    first_rejection_diagnostic: Any | None = None,
    reviewer_cache_root: Path | None = None,
    independent_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Run, strictly evaluate, persist, and return one official task result."""
    if first_rejection_diagnostic is not None and (
        config.mode != "shadow"
        or config.pairing_mode != "replay"
        or config.verification_budget != 1
    ):
        raise ValueError("first-rejection diagnostic requires paired Shadow B1")
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from .orchestrator import ShadowTau2Orchestrator
    from .participants import build_native_agent, build_native_user

    if config.mode == "baseline":
        if candidate_test_suite is not None:
            raise ValueError("candidate policy tests are valid only in shadow mode")
    elif config.candidate_test_suite_sha256 is None:
        if candidate_test_suite is not None:
            raise ValueError(
                "suite supplied while config disables candidate policy tests"
            )
    else:
        if not isinstance(candidate_test_suite, PolicyTestSuite):
            raise ValueError(
                "shadow config enables candidate policy tests but suite is missing"
            )
        if (
            candidate_test_suite.benchmark != "tau2"
            or candidate_test_suite.domain != config.domain
            or candidate_test_suite.compiler_model != config.producer_model
            or (candidate_test_suite.suite_sha256 != config.candidate_test_suite_sha256)
        ):
            raise ValueError("candidate policy-test suite identity mismatch")
    official_task = load_pilot_tasks(
        (task.id,), domain=config.domain, task_split=config.task_split
    )[0]
    task_fingerprint = _task_fingerprint(task)
    official_task_fingerprint = _task_fingerprint(official_task)
    if task_fingerprint != official_task_fingerprint:
        raise ValueError("run_task received a modified or non-official task")
    path = result_path(output_root, config, task.id)
    provenance = data_provenance(config.domain, config.task_split)
    provenance_fingerprint = _canonical_sha256(provenance)
    implementation = implementation_provenance()
    implementation_fingerprint = implementation["fingerprint"]
    artifact_identity = {
        "schema": RUNNER_SCHEMA,
        "benchmark": "tau2",
        "domain": config.domain,
        "task_split": config.task_split,
        "sample_manifest_fingerprint": config.sample_manifest_fingerprint,
        "task_id": task.id,
        "reviewer_variant": config.reviewer_variant,
        "reviewer_system_prompt_sha256": config.reviewer_system_prompt_sha256,
        "reviewer_enabled": config.mode == "shadow",
        "paired_cache_source_pair_run_id": config.paired_cache_source_pair_run_id,
        "paired_cache_source_manifest_fingerprint": config.paired_cache_source_manifest_fingerprint,
        "config_fingerprint": config.fingerprint,
        "data_provenance_fingerprint": provenance_fingerprint,
        "implementation_fingerprint": implementation_fingerprint,
        "official_task_fingerprint": official_task_fingerprint,
    }
    if first_rejection_diagnostic is not None:
        artifact_identity["first_rejection_diagnostic"] = (
            first_rejection_diagnostic.metadata
        )
    if resume and path.exists():
        if not path.is_file():
            raise ResultArtifactConflictError(
                "result artifact path exists but is not a regular file"
            )
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ResultArtifactConflictError(
                "existing result artifact is unreadable"
            ) from None
        if not isinstance(prior, dict):
            raise ResultArtifactConflictError(
                "existing result artifact is not a JSON object"
            )
        same_identity = all(
            (prior.get(key) == value for key, value in artifact_identity.items())
        )
        if same_identity and prior.get("status") == "complete":
            return prior
        if not same_identity or prior.get("status") not in {"running", "failed"}:
            raise ResultArtifactConflictError(
                "existing result artifact belongs to a different experiment; use a new output root or explicitly disable resume"
            )
    atomic_json_dump(path, {"status": "running", **artifact_identity})
    environment = registry.get_env_constructor(config.domain)()
    producer_backend = _backend(
        config.producer_model,
        config,
        output_root=output_root,
        task_id=task.id,
        participant_role="producer",
    )
    user_backend = _backend(
        config.user_model,
        config,
        output_root=output_root,
        task_id=task.id,
        participant_role="user",
    )
    if independent_cache_root is not None:
        if config.pairing_mode != "independent":
            raise ValueError("independent cache requires independent pairing mode")
        from workflow.experiment_replay import OccurrenceCacheBackend

        producer_backend = OccurrenceCacheBackend(
            producer_backend, independent_cache_root / "producer"
        )
        user_backend = OccurrenceCacheBackend(
            user_backend, independent_cache_root / "user"
        )
    agent = build_native_agent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        backend=producer_backend,
    )
    user = build_native_user(
        instructions=str(task.user_scenario),
        tools=_user_tools(environment, task),
        backend=user_backend,
    )
    common = {
        "domain": config.domain,
        "agent": agent,
        "user": user,
        "environment": environment,
        "task": task,
        "max_steps": config.max_steps,
        "max_errors": config.max_errors,
        "seed": config.seed,
        "solo_mode": False,
        "validate_communication": config.validate_communication,
        "timeout": config.episode_timeout_seconds,
    }
    if config.mode == "shadow":
        reviewer_runtime_model = config.judge_model
        if reviewer_runtime_model is None:
            raise RuntimeError("shadow reviewer model identity is missing")

        def cached_reviewer_backend(model):
            backend = _backend(
                model,
                config,
                output_root=output_root,
                task_id=task.id,
                participant_role="judge",
                judge=True,
            )
            return (
                ReplayCacheBackend(backend, reviewer_cache_root)
                if reviewer_cache_root is not None
                else backend
            )

        reviewer_factory = SemanticReviewerFactory(
            cached_reviewer_backend,
            model=reviewer_runtime_model,
            config=SemanticReviewerConfig(
                seed=config.seed,
                system_prompt=reviewer_prompt(config.reviewer_variant),
                soft_label_output=config.reviewer_soft_label_output,
                candidate_effect_visibility=config.reviewer_candidate_effect_visibility,
            ),
            sample_seed_mode=config.reviewer_seed_mode,
        )
        orchestrator: Any = ShadowTau2Orchestrator(
            reviewer_factory=first_rejection_diagnostic.factory
            if first_rejection_diagnostic is not None
            else reviewer_factory,
            first_rejection_diagnostic=first_rejection_diagnostic,
            verification_budget=config.verification_budget,
            probability_aggregation=config.reviewer_probability_aggregation,
            probability_threshold=config.reviewer_probability_threshold,
            probability_early_stop=config.reviewer_probability_early_stop,
            include_tool_schemas_in_evidence=config.include_tool_schemas_in_evidence,
            candidate_test_suite=candidate_test_suite,
            on_first_rejection=lambda: (
                producer_backend.release_replay_only(),
                user_backend.release_replay_only(),
            )
            if config.pairing_mode == "replay"
            else None,
            **common,
        )
    else:
        orchestrator = Orchestrator(**common)
    started = time.perf_counter()
    producer_simulation = orchestrator.run()
    if first_rejection_diagnostic is not None:
        first_rejection_diagnostic.assert_complete()
    elapsed_s = time.perf_counter() - started
    if config.mode == "shadow":
        canonical_messages = _normalise_turn_indices(
            orchestrator.canonical_trajectory()
        )
    else:
        canonical_messages = _normalise_turn_indices(producer_simulation.messages or ())
    orchestrator.validate_message_history(list(canonical_messages))
    canonical_simulation = producer_simulation.model_copy(deep=True)
    canonical_simulation.messages = canonical_messages
    canonical_evaluation = _evaluate(
        canonical_simulation,
        task,
        config=config,
        output_root=output_root,
        task_fingerprint=official_task_fingerprint,
        strict_replay=True,
    )
    canonical_reward = canonical_evaluation.reward_info
    canonical_simulation.reward_info = canonical_reward
    counterfactual_reward = None
    counterfactual_evaluation_audit = None
    counterfactual_evaluation_error = None
    if config.mode == "shadow" and orchestrator.rejected_call_ids:
        all_proposals = producer_simulation.model_copy(deep=True)
        all_proposals.messages = _normalise_turn_indices(
            producer_simulation.messages or ()
        )
        try:
            counterfactual_evaluation = _evaluate(
                all_proposals,
                task,
                config=config,
                output_root=output_root,
                task_fingerprint=official_task_fingerprint,
                strict_replay=False,
            )
            counterfactual_reward = counterfactual_evaluation.reward_info
            counterfactual_evaluation_audit = counterfactual_evaluation.audit
        except Exception as exc:
            counterfactual_evaluation_error = {"error_type": type(exc).__name__}
    producer_messages = producer_simulation.messages or []
    canonical_mutations = _mutating_calls(canonical_messages, orchestrator.environment)
    producer_mutations = _mutating_calls(producer_messages, orchestrator.environment)
    gate_events = (
        copy.deepcopy(orchestrator.gate_events) if config.mode == "shadow" else []
    )
    rejected_batches = sum((event["outcome"] == "rejected" for event in gate_events))
    published_batches = sum((event["outcome"] == "published" for event in gate_events))
    official_reward = float(canonical_reward.reward)
    counterfactual_value = (
        float(counterfactual_reward.reward)
        if counterfactual_reward is not None
        else None
    )
    metrics = {
        **_message_metrics(producer_messages),
        "elapsed_s": round(elapsed_s, 6),
        "message_count_producer": len(producer_messages),
        "message_count_canonical": len(canonical_messages),
        "mutation_capable_batches_proposed": len(producer_mutations),
        "mutation_capable_batches_published": len(canonical_mutations),
        "shadow_reviewed_batches": len(gate_events),
        "shadow_judge_calls": 0
        if first_rejection_diagnostic is not None
        else sum((int(event.get("verification_budget", 1)) for event in gate_events)),
        **{},
        "candidate_policy_test_reports": sum(
            ("candidate_policy_test_report" in event for event in gate_events)
        ),
        "candidate_policy_tests_failed": sum(
            (
                event.get("candidate_policy_test_report", {})
                .get("summary", {})
                .get("failed", 0)
                for event in gate_events
            )
        ),
        "candidate_policy_tests_error": sum(
            (
                event.get("candidate_policy_test_report", {})
                .get("summary", {})
                .get("error", 0)
                for event in gate_events
            )
        ),
        "shadow_rejected_batches": rejected_batches,
        "shadow_published_batches": published_batches,
        "shadow_review_seconds": round(
            sum((float(event["review_seconds"]) for event in gate_events)), 6
        ),
        "nl_evaluator_api_calls": int(
            canonical_evaluation.audit["nl_evaluation"].get("api_calls", 0)
        ),
        "nl_evaluator_cache_hits": int(
            canonical_evaluation.audit["nl_evaluation"].get("cache_hit", False)
        ),
        "nl_evaluator_prompt_tokens": int(
            canonical_evaluation.audit["nl_evaluation"].get("prompt_tokens", 0)
        ),
        "nl_evaluator_completion_tokens": int(
            canonical_evaluation.audit["nl_evaluation"].get("completion_tokens", 0)
        ),
        "nl_evaluator_total_tokens": int(
            canonical_evaluation.audit["nl_evaluation"].get("total_tokens", 0)
        ),
        "nl_evaluator_seconds": round(
            float(canonical_evaluation.audit["nl_evaluation"].get("latency_s", 0.0)), 6
        ),
        "shadow_success_with_rejection": bool(
            rejected_batches and official_reward == 1.0
        ),
        "canonical_beats_all_proposals_replay": bool(
            rejected_batches
            and counterfactual_value is not None
            and (official_reward > counterfactual_value)
        ),
    }
    payload = {
        "status": "complete",
        "schema": RUNNER_SCHEMA,
        "benchmark": "tau2",
        "domain": config.domain,
        "task_split": config.task_split,
        "sample_manifest_fingerprint": config.sample_manifest_fingerprint,
        "task_id": task.id,
        "official_task_fingerprint": official_task_fingerprint,
        "mode": config.mode,
        "reviewer_variant": config.reviewer_variant,
        "reviewer_system_prompt_sha256": config.reviewer_system_prompt_sha256,
        "reviewer_enabled": config.mode == "shadow",
        "verification_budget": config.verification_budget,
        "reviewer_temperature": config.reviewer_temperature,
        "reviewer_seed_mode": config.reviewer_seed_mode,
        "reviewer_candidate_effect_visibility": config.reviewer_candidate_effect_visibility,
        **{},
        **(
            {"reviewer_soft_label_output": True}
            if config.reviewer_soft_label_output
            else {}
        ),
        **(
            {
                "reviewer_probability_aggregation": config.reviewer_probability_aggregation
            }
            if config.reviewer_probability_aggregation != "vote_majority"
            else {}
        ),
        **(
            {"reviewer_probability_threshold": config.reviewer_probability_threshold}
            if config.reviewer_probability_threshold != 0.5
            else {}
        ),
        **(
            {"reviewer_probability_early_stop": True}
            if config.reviewer_probability_early_stop
            else {}
        ),
        "paired_cache_source_pair_run_id": config.paired_cache_source_pair_run_id,
        "paired_cache_source_manifest_fingerprint": config.paired_cache_source_manifest_fingerprint,
        "producer_model": config.producer_model,
        "judge_model": config.judge_model if config.mode == "shadow" else None,
        "nl_evaluator_model": config.nl_evaluator_model,
        "user_model": config.user_model,
        "seed": config.seed,
        "config": config.artifact_config,
        "config_fingerprint": config.fingerprint,
        "pair_run_id": config.pair_run_id,
        "pairing_mode": config.pairing_mode,
        "pair_common_fingerprint": config.pair_common_fingerprint,
        "data_provenance": provenance,
        **(
            {"first_rejection_diagnostic": first_rejection_diagnostic.metadata}
            if first_rejection_diagnostic is not None
            else {}
        ),
        "data_provenance_fingerprint": provenance_fingerprint,
        "implementation_provenance": implementation,
        "implementation_fingerprint": implementation_fingerprint,
        "candidate_policy_tests": {
            "enabled": True,
            "suite_sha256": candidate_test_suite.suite_sha256,
            "compiler_model": candidate_test_suite.compiler_model,
            "policy_sha256": candidate_test_suite.policy_sha256,
            "candidate_schema_sha256": candidate_test_suite.candidate_schema_sha256,
            "test_count": len(candidate_test_suite.tests),
        }
        if candidate_test_suite is not None
        else {"enabled": False},
        "code_provenance": {
            "implementation_git_head": _git_head(REPO_ROOT),
            "tau2_git_head": _git_head(TAU2_ROOT),
        },
        "integrity": {
            "official_task": True,
            "official_environment": True,
            "official_evaluator": canonical_evaluation.audit["leaderboard_official"],
            "official_structural_evaluators": True,
            "official_nl_assertion_evaluator": not canonical_evaluation.audit[
                "task_requires_nl_assertion"
            ],
            "leaderboard_official_tsr": canonical_evaluation.audit[
                "leaderboard_official"
            ],
            "strict_canonical_replay": True,
            "surrogate_or_mock_score": canonical_evaluation.audit[
                "task_requires_nl_assertion"
            ],
            "gold_available_to_producer": False,
            "gold_available_to_shadow_judge": False,
            "gold_available_to_offline_nl_evaluator": canonical_evaluation.audit[
                "task_requires_nl_assertion"
            ],
            "offline_nl_evaluator_runs_post_episode_only": True,
            "judge_online_inputs": [
                "domain_policy",
                "canonical_public_history",
                "current_assistant_tool_batch",
                "candidate_tool_results",
                "candidate_structural_state_diff",
                *(
                    ["candidate_policy_test_report"]
                    if candidate_test_suite is not None
                    else []
                ),
            ],
            "counterfactual_all_proposals_is_official_tsr": False,
            "paired_replay_cache_enabled": config.pairing_mode != "independent",
            "paired_replay_only_before_first_rejection": config.pairing_mode
            == "replay",
            "paired_replay_cache_stores_prompts": False,
        },
        "evaluation_audit": canonical_evaluation.audit,
        "harness": {
            "producer": "official tau2 LLMAgent state machine + native DashScope backend",
            "user": "official tau2 UserSimulator state machine + native DashScope backend",
            "orchestrator": "workflow.tau2.ShadowTau2Orchestrator"
            if config.mode == "shadow"
            else "tau2.orchestrator.orchestrator.Orchestrator",
            "communication": "official half-duplex; text XOR native tool calls",
            "shadow_scope": "whole assistant tool batch if any call mutates state",
            "pure_reads_gated": False,
            "user_tool_calls_gated": False,
            "producer_system_prompt_sha256": _canonical_sha256(agent.system_prompt),
            "user_system_prompt_sha256": _canonical_sha256(user.system_prompt),
            "producer_tool_schema_sha256": _canonical_sha256(
                [tool.openai_schema for tool in agent.tools]
            ),
        },
        "official_reward": official_reward,
        "protocol_reward": official_reward,
        "score_protocol": canonical_evaluation.audit["score_protocol"],
        "leaderboard_official": canonical_evaluation.audit["leaderboard_official"],
        "official_reward_label_valid": canonical_evaluation.audit[
            "leaderboard_official"
        ],
        "official_reward_info": canonical_reward.model_dump(mode="json"),
        "counterfactual_all_proposals_reward": counterfactual_value,
        "counterfactual_all_proposals_reward_info": counterfactual_reward.model_dump(
            mode="json"
        )
        if counterfactual_reward is not None
        else None,
        "counterfactual_all_proposals_evaluation_audit": counterfactual_evaluation_audit,
        "counterfactual_all_proposals_evaluation_error": counterfactual_evaluation_error,
        "metrics": metrics,
        "gate_events": gate_events,
        "canonical_mutation_capable_batches": canonical_mutations,
        "producer_mutation_capable_batches": producer_mutations,
        "canonical_simulation": canonical_simulation.model_dump(mode="json"),
        "producer_simulation": producer_simulation.model_dump(mode="json"),
    }
    atomic_json_dump(path, payload)
    return payload


def summarize_cell(
    output_root: Path,
    config: Tau2PilotConfig,
    *,
    expected_task_ids: Sequence[str] = PILOT_TASK_IDS,
) -> dict[str, Any]:
    current_data_fingerprint = _canonical_sha256(
        data_provenance(config.domain, config.task_split)
    )
    current_implementation_fingerprint = implementation_provenance()["fingerprint"]
    rows: list[dict[str, Any]] = []
    for task_id in expected_task_ids:
        path = result_path(output_root, config, str(task_id))
        if not path.is_file():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            row.get("status") == "complete"
            and row.get("schema") == RUNNER_SCHEMA
            and (row.get("domain") == config.domain)
            and (row.get("task_split") == config.task_split)
            and (
                row.get("sample_manifest_fingerprint")
                == config.sample_manifest_fingerprint
            )
            and (row.get("reviewer_variant") == config.reviewer_variant)
            and (
                row.get("reviewer_system_prompt_sha256")
                == config.reviewer_system_prompt_sha256
            )
            and (row.get("reviewer_enabled") == (config.mode == "shadow"))
            and (
                row.get("paired_cache_source_pair_run_id")
                == config.paired_cache_source_pair_run_id
            )
            and (
                row.get("paired_cache_source_manifest_fingerprint")
                == config.paired_cache_source_manifest_fingerprint
            )
            and (row.get("config_fingerprint") == config.fingerprint)
            and (row.get("data_provenance_fingerprint") == current_data_fingerprint)
            and (
                row.get("implementation_fingerprint")
                == current_implementation_fingerprint
            )
        ):
            rows.append(row)
    rewards = [
        float(row.get("protocol_reward", row["official_reward"])) for row in rows
    ]
    score_protocols = sorted(
        {
            str(row["score_protocol"])
            for row in rows
            if isinstance(row.get("score_protocol"), str)
        }
    )
    leaderboard_official = (
        all((row.get("leaderboard_official") is True for row in rows)) if rows else None
    )
    rejected = sum((int(row["metrics"]["shadow_rejected_batches"]) for row in rows))
    summary = {
        "status": "complete"
        if len(rows) == len(tuple(expected_task_ids))
        else "partial",
        "schema": RUNNER_SCHEMA,
        "domain": config.domain,
        "task_split": config.task_split,
        "sample_manifest_fingerprint": config.sample_manifest_fingerprint,
        "mode": config.mode,
        "reviewer_variant": config.reviewer_variant,
        "reviewer_system_prompt_sha256": config.reviewer_system_prompt_sha256,
        "reviewer_enabled": config.mode == "shadow",
        "verification_budget": config.verification_budget,
        "reviewer_temperature": config.reviewer_temperature,
        "reviewer_seed_mode": config.reviewer_seed_mode,
        **{},
        **(
            {"reviewer_soft_label_output": True}
            if config.reviewer_soft_label_output
            else {}
        ),
        **(
            {
                "reviewer_probability_aggregation": config.reviewer_probability_aggregation
            }
            if config.reviewer_probability_aggregation != "vote_majority"
            else {}
        ),
        **(
            {"reviewer_probability_threshold": config.reviewer_probability_threshold}
            if config.reviewer_probability_threshold != 0.5
            else {}
        ),
        **(
            {"reviewer_probability_early_stop": True}
            if config.reviewer_probability_early_stop
            else {}
        ),
        "paired_cache_source_pair_run_id": config.paired_cache_source_pair_run_id,
        "paired_cache_source_manifest_fingerprint": config.paired_cache_source_manifest_fingerprint,
        "producer_model": config.producer_model,
        "judge_model": config.judge_model if config.mode == "shadow" else None,
        "nl_evaluator_model": config.nl_evaluator_model,
        "seed": config.seed,
        "pair_run_id": config.pair_run_id,
        "pairing_mode": config.pairing_mode,
        "pair_common_fingerprint": config.pair_common_fingerprint,
        "tasks_planned": len(tuple(expected_task_ids)),
        "tasks_complete": len(rows),
        "score_protocol": score_protocols[0]
        if len(score_protocols) == 1
        else "mixed"
        if score_protocols
        else None,
        "score_protocols": score_protocols,
        "leaderboard_official": leaderboard_official,
        "official_tsr_label_valid": leaderboard_official,
        "protocol_successes": sum(rewards),
        "protocol_tsr": sum(rewards) / len(rewards) if rewards else None,
        "official_successes": sum(rewards) if leaderboard_official else None,
        "official_tsr": sum(rewards) / len(rewards)
        if rewards and leaderboard_official
        else None,
        "shadow_rejected_batches": rejected,
        "tasks_with_rejection": sum(
            (int(row["metrics"]["shadow_rejected_batches"] > 0) for row in rows)
        ),
        "shadow_successes_with_rejection": sum(
            (bool(row["metrics"]["shadow_success_with_rejection"]) for row in rows)
        ),
        "tasks_where_canonical_beats_all_proposals_replay": sum(
            (
                bool(row["metrics"]["canonical_beats_all_proposals_replay"])
                for row in rows
            )
        ),
        "agent_api_calls": sum((row["metrics"]["agent_api_calls"] for row in rows)),
        "user_api_calls": sum((row["metrics"]["user_api_calls"] for row in rows)),
        "agent_replay_cache_hits": sum(
            (row["metrics"]["agent_replay_cache_hits"] for row in rows)
        ),
        "user_replay_cache_hits": sum(
            (row["metrics"]["user_replay_cache_hits"] for row in rows)
        ),
        "judge_api_calls": sum((row["metrics"]["shadow_judge_calls"] for row in rows)),
        **{},
        "reviewed_batches": sum(
            (
                row["metrics"].get(
                    "shadow_reviewed_batches", row["metrics"]["shadow_judge_calls"]
                )
                for row in rows
            )
        ),
        "nl_evaluator_api_calls": sum(
            (int(row["metrics"].get("nl_evaluator_api_calls", 0)) for row in rows)
        ),
        "nl_evaluator_cache_hits": sum(
            (int(row["metrics"].get("nl_evaluator_cache_hits", 0)) for row in rows)
        ),
        "elapsed_s": round(sum((row["metrics"]["elapsed_s"] for row in rows)), 6),
        "elapsed_s_is_deployment_latency": config.pairing_mode != "replay",
        "config_fingerprint": config.fingerprint,
        "data_provenance_fingerprint": current_data_fingerprint,
        "implementation_fingerprint": current_implementation_fingerprint,
    }
    return summary


def _model_completion_trace(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    trace: list[dict[str, Any]] = []
    missing_digest_indices: list[int] = []
    messages = row.get("producer_simulation", {}).get("messages") or []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in (
            "assistant",
            "user",
        ):
            continue
        raw = message.get("raw_data")
        generated = (
            isinstance(raw, dict)
            or isinstance(message.get("usage"), dict)
            or message.get("generation_time_seconds") is not None
        )
        if not generated:
            continue
        request_digest = (
            raw.get("replay_request_sha256") if isinstance(raw, dict) else None
        )
        completion_digest = (
            raw.get("replay_completion_sha256") if isinstance(raw, dict) else None
        )
        if (
            not isinstance(request_digest, str)
            or re.fullmatch("[0-9a-f]{64}", request_digest) is None
            or (not isinstance(completion_digest, str))
            or (re.fullmatch("[0-9a-f]{64}", completion_digest) is None)
        ):
            missing_digest_indices.append(message_index)
            continue
        calls = message.get("tool_calls") or []
        trace.append(
            {
                "message_index": message_index,
                "role": message["role"],
                "request_sha256": request_digest,
                "completion_sha256": completion_digest,
                "cache_hit": raw.get("replay_cache_hit") is True,
                "tool_call_ids": [
                    call.get("id")
                    for call in calls
                    if isinstance(call, dict) and isinstance(call.get("id"), str)
                ],
            }
        )
    return (trace, missing_digest_indices)


def _canonical_semantic_digest(row: dict[str, Any]) -> str:
    """Digest canonical behavior while excluding timing/provider metadata."""
    messages = row.get("canonical_simulation", {}).get("messages") or []
    projected: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            projected.append({"invalid_message_type": type(message).__name__})
            continue
        item: dict[str, Any] = {"role": message.get("role")}
        if message.get("content") is not None:
            item["content"] = message.get("content")
        if message.get("id") is not None:
            item["id"] = message.get("id")
        if message.get("requestor") is not None:
            item["requestor"] = message.get("requestor")
        if message.get("error") is not None:
            item["error"] = message.get("error")
        calls = message.get("tool_calls")
        if calls is not None:
            item["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "requestor": call.get("requestor"),
                }
                for call in calls
                if isinstance(call, dict)
            ]
        nested = message.get("tool_messages")
        if nested is not None:
            item["tool_messages"] = [
                {
                    "id": result.get("id"),
                    "role": result.get("role"),
                    "requestor": result.get("requestor"),
                    "content": result.get("content"),
                    "error": result.get("error"),
                }
                for result in nested
                if isinstance(result, dict)
            ]
        projected.append(item)
    return _canonical_sha256(
        {
            "messages": projected,
            "termination_reason": row.get("canonical_simulation", {}).get(
                "termination_reason"
            ),
        }
    )


def _pair_task_row(
    task_id: str, baseline: dict[str, Any], shadow: dict[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    for name, row, expected_mode, expected_pairing in (
        ("baseline", baseline, "baseline", "record"),
        ("shadow", shadow, "shadow", "replay"),
    ):
        if row.get("status") != "complete":
            reasons.append(f"{name}_incomplete")
        if row.get("mode") != expected_mode:
            reasons.append(f"{name}_mode_mismatch")
        if row.get("pairing_mode") != expected_pairing:
            reasons.append(f"{name}_pairing_mode_mismatch")
    for field in (
        "pair_run_id",
        "pair_common_fingerprint",
        "data_provenance_fingerprint",
        "implementation_fingerprint",
    ):
        if not baseline.get(field) or baseline.get(field) != shadow.get(field):
            reasons.append(f"{field}_mismatch")
    baseline_score_protocol = baseline.get("score_protocol")
    shadow_score_protocol = shadow.get("score_protocol")
    if (
        not isinstance(baseline_score_protocol, str)
        or baseline_score_protocol != shadow_score_protocol
    ):
        reasons.append("score_protocol_mismatch")
    baseline_leaderboard_official = baseline.get("leaderboard_official") is True
    shadow_leaderboard_official = shadow.get("leaderboard_official") is True
    if baseline_leaderboard_official != shadow_leaderboard_official:
        reasons.append("leaderboard_official_mismatch")
    baseline_trace, baseline_missing_digests = _model_completion_trace(baseline)
    shadow_trace, shadow_missing_digests = _model_completion_trace(shadow)
    if baseline_missing_digests:
        reasons.append("baseline_generated_messages_missing_digests")
    if shadow_missing_digests:
        reasons.append("shadow_generated_messages_missing_digests")
    for name, row, trace, missing in (
        ("baseline", baseline, baseline_trace, baseline_missing_digests),
        ("shadow", shadow, shadow_trace, shadow_missing_digests),
    ):
        metrics = row.get("metrics", {})
        expected_invocations = int(metrics.get("agent_model_invocations", 0)) + int(
            metrics.get("user_model_invocations", 0)
        )
        if len(trace) + len(missing) != expected_invocations:
            reasons.append(f"{name}_model_invocation_count_mismatch")
    rejected_events = [
        event
        for event in shadow.get("gate_events", [])
        if isinstance(event, dict) and event.get("outcome") == "rejected"
    ]
    first_rejected_ids = (
        list(rejected_events[0].get("tool_call_ids") or []) if rejected_events else []
    )
    prefix_length = len(shadow_trace)
    if first_rejected_ids:
        rejected_tuple = tuple(first_rejected_ids)
        matching_indices = [
            index
            for index, item in enumerate(shadow_trace)
            if tuple(item["tool_call_ids"]) == rejected_tuple
        ]
        if len(matching_indices) != 1:
            reasons.append("first_rejection_not_unique_in_shadow_trace")
            prefix_length = 0
        else:
            prefix_length = matching_indices[0] + 1
    baseline_prefix = baseline_trace[:prefix_length]
    shadow_prefix = shadow_trace[:prefix_length]
    digest_fields = ("role", "request_sha256", "completion_sha256")
    prefix_exact = (
        prefix_length > 0
        and len(baseline_prefix) == prefix_length
        and (len(shadow_prefix) == prefix_length)
        and all((item["cache_hit"] for item in shadow_prefix))
        and all(
            (
                tuple((left[field] for field in digest_fields))
                == tuple((right[field] for field in digest_fields))
                for left, right in zip(baseline_prefix, shadow_prefix, strict=True)
            )
        )
    )
    if not first_rejected_ids:
        prefix_exact = (
            prefix_exact
            and len(baseline_trace) == len(shadow_trace)
            and (prefix_length == len(baseline_trace))
        )
    if not prefix_exact:
        reasons.append("preintervention_prefix_not_exact")
    baseline_canonical_digest = _canonical_semantic_digest(baseline)
    shadow_canonical_digest = _canonical_semantic_digest(shadow)
    baseline_reward_info_digest = _canonical_sha256(
        baseline.get("official_reward_info")
    )
    shadow_reward_info_digest = _canonical_sha256(shadow.get("official_reward_info"))
    no_intervention_semantics_equal = (
        baseline_canonical_digest == shadow_canonical_digest
        and baseline_reward_info_digest == shadow_reward_info_digest
        and (baseline.get("official_reward") == shadow.get("official_reward"))
    )
    if not first_rejected_ids and (not no_intervention_semantics_equal):
        reasons.append("no_intervention_canonical_divergence")
    baseline_mutations = baseline.get("canonical_mutation_capable_batches") or []
    rejected_tuple = tuple(first_rejected_ids)
    matching_baseline_mutation_indices = (
        [
            index
            for index, batch in enumerate(baseline_mutations)
            if tuple(batch.get("tool_call_ids") or []) == rejected_tuple
        ]
        if rejected_tuple
        else []
    )
    first_rejected_proposal_in_baseline_mutation_capable_batch = (
        len(matching_baseline_mutation_indices) == 1
    )
    baseline_mutations_after = (
        len(baseline_mutations) - matching_baseline_mutation_indices[0] - 1
        if first_rejected_proposal_in_baseline_mutation_capable_batch
        else None
    )
    baseline_reward = float(
        baseline.get("protocol_reward", baseline.get("official_reward", 0.0))
    )
    shadow_reward = float(
        shadow.get("protocol_reward", shadow.get("official_reward", 0.0))
    )
    valid = not reasons
    return {
        "task_id": task_id,
        "valid": valid,
        "invalid_reasons": reasons,
        "baseline_reward": baseline_reward,
        "shadow_reward": shadow_reward,
        "score_protocol": baseline_score_protocol,
        "leaderboard_official": baseline_leaderboard_official,
        "outcome": f"{int(baseline_reward)}->{int(shadow_reward)}",
        "first_rejection_present": bool(first_rejected_ids),
        "first_rejected_tool_call_ids": first_rejected_ids,
        "first_rejected_proposal_in_baseline_mutation_capable_batch": first_rejected_proposal_in_baseline_mutation_capable_batch,
        "preintervention_prefix_exact": prefix_exact,
        "preintervention_matched_model_invocations": prefix_length
        if prefix_exact
        else 0,
        "baseline_generated_messages_missing_digests": baseline_missing_digests,
        "shadow_generated_messages_missing_digests": shadow_missing_digests,
        "baseline_canonical_semantic_sha256": baseline_canonical_digest,
        "shadow_canonical_semantic_sha256": shadow_canonical_digest,
        "no_intervention_semantics_equal": no_intervention_semantics_equal
        if not first_rejected_ids
        else None,
        "baseline_mutation_capable_batches_after_first_rejected_proposal": baseline_mutations_after,
        "shadow_rejected_batches": int(
            shadow.get("metrics", {}).get("shadow_rejected_batches", 0)
        ),
        "shadow_success_with_rejection": bool(
            first_rejected_ids and shadow_reward == 1.0
        ),
        "paired_fail_to_success_after_rejection": bool(
            valid
            and first_rejected_ids
            and (baseline_reward == 0.0)
            and (shadow_reward == 1.0)
        ),
        "paired_success_to_failure_after_rejection": bool(
            valid
            and first_rejected_ids
            and (baseline_reward == 1.0)
            and (shadow_reward == 0.0)
        ),
    }


def summarize_paired(
    output_root: Path,
    baseline_config: Tau2PilotConfig,
    shadow_config: Tau2PilotConfig,
    *,
    expected_task_ids: Sequence[str] = PILOT_TASK_IDS,
) -> dict[str, Any]:
    """Join experiment arms task-by-task and validate exact causal prefixes."""
    if baseline_config.pairing_mode != "record":
        raise ValueError("baseline_config must use pairing_mode=record")
    if shadow_config.pairing_mode != "replay":
        raise ValueError("shadow_config must use pairing_mode=replay")
    if (
        baseline_config.pair_run_id != shadow_config.pair_run_id
        or baseline_config.pair_common_fingerprint
        != shadow_config.pair_common_fingerprint
    ):
        raise ValueError("paired configs do not share one experiment identity")
    requested_tuple = tuple((str(item) for item in expected_task_ids))
    if (
        baseline_config.paired_task_ids != requested_tuple
        or shadow_config.paired_task_ids != requested_tuple
    ):
        raise ValueError("paired summary task set differs from immutable config")
    pair_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    current_data_fingerprint = _canonical_sha256(
        data_provenance(baseline_config.domain, baseline_config.task_split)
    )
    current_implementation_fingerprint = implementation_provenance()["fingerprint"]
    for task_id in requested_tuple:
        baseline_path = result_path(output_root, baseline_config, task_id)
        shadow_path = result_path(output_root, shadow_config, task_id)
        if not baseline_path.is_file() or not shadow_path.is_file():
            missing.append(task_id)
            continue
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(task_id)
            continue
        pair_row = _pair_task_row(task_id, baseline, shadow)
        for name, row, expected_config in (
            ("baseline", baseline, baseline_config),
            ("shadow", shadow, shadow_config),
        ):
            if row.get("schema") != RUNNER_SCHEMA:
                pair_row["invalid_reasons"].append(f"{name}_schema_mismatch")
            if row.get("domain") != expected_config.domain:
                pair_row["invalid_reasons"].append(f"{name}_domain_mismatch")
            if row.get("task_split") != expected_config.task_split:
                pair_row["invalid_reasons"].append(f"{name}_task_split_mismatch")
            if (
                row.get("sample_manifest_fingerprint")
                != expected_config.sample_manifest_fingerprint
            ):
                pair_row["invalid_reasons"].append(
                    f"{name}_sample_manifest_fingerprint_mismatch"
                )
            if row.get("config_fingerprint") != expected_config.fingerprint:
                pair_row["invalid_reasons"].append(
                    f"{name}_config_fingerprint_mismatch"
                )
            if row.get("reviewer_variant") != expected_config.reviewer_variant:
                pair_row["invalid_reasons"].append(f"{name}_reviewer_variant_mismatch")
            if (
                row.get("reviewer_system_prompt_sha256")
                != expected_config.reviewer_system_prompt_sha256
            ):
                pair_row["invalid_reasons"].append(
                    f"{name}_reviewer_system_prompt_sha256_mismatch"
                )
            if row.get("reviewer_enabled") != (expected_config.mode == "shadow"):
                pair_row["invalid_reasons"].append(f"{name}_reviewer_enabled_mismatch")
            if (
                row.get("paired_cache_source_pair_run_id")
                != expected_config.paired_cache_source_pair_run_id
            ):
                pair_row["invalid_reasons"].append(
                    f"{name}_paired_cache_source_pair_run_id_mismatch"
                )
            if (
                row.get("paired_cache_source_manifest_fingerprint")
                != expected_config.paired_cache_source_manifest_fingerprint
            ):
                pair_row["invalid_reasons"].append(
                    f"{name}_paired_cache_source_manifest_fingerprint_mismatch"
                )
            if row.get("data_provenance_fingerprint") != current_data_fingerprint:
                pair_row["invalid_reasons"].append(f"{name}_stale_data_fingerprint")
            if (
                row.get("implementation_fingerprint")
                != current_implementation_fingerprint
            ):
                pair_row["invalid_reasons"].append(
                    f"{name}_stale_implementation_fingerprint"
                )
        if pair_row["invalid_reasons"]:
            pair_row["valid"] = False
            pair_row["paired_fail_to_success_after_rejection"] = False
            pair_row["paired_success_to_failure_after_rejection"] = False
        pair_rows.append(pair_row)
    valid_rows = [row for row in pair_rows if row["valid"]]
    baseline_rewards = [row["baseline_reward"] for row in valid_rows]
    shadow_rewards = [row["shadow_reward"] for row in valid_rows]
    quadrants = {
        outcome: sum((row["outcome"] == outcome for row in valid_rows))
        for outcome in ("0->0", "0->1", "1->0", "1->1")
    }
    planned = len(requested_tuple)
    status = (
        "complete"
        if len(valid_rows) == planned and (not missing)
        else "invalid"
        if pair_rows and any((not row["valid"] for row in pair_rows))
        else "partial"
    )
    baseline_tsr = (
        sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else None
    )
    shadow_tsr = sum(shadow_rewards) / len(shadow_rewards) if shadow_rewards else None
    score_protocols = sorted(
        {
            row["score_protocol"]
            for row in valid_rows
            if isinstance(row.get("score_protocol"), str)
        }
    )
    leaderboard_official = (
        all((row["leaderboard_official"] for row in valid_rows)) if valid_rows else None
    )
    return {
        "status": status,
        "schema": "shadow-tau2-paired-comparison-v4-score-protocol",
        "pair_run_id": baseline_config.pair_run_id,
        "pair_common_fingerprint": baseline_config.pair_common_fingerprint,
        "domain": baseline_config.domain,
        "task_split": baseline_config.task_split,
        "sample_manifest_fingerprint": baseline_config.sample_manifest_fingerprint,
        "reviewer_variant": shadow_config.reviewer_variant,
        "verification_budget": shadow_config.verification_budget,
        **{},
        **(
            {"reviewer_soft_label_output": True}
            if shadow_config.reviewer_soft_label_output
            else {}
        ),
        **(
            {
                "reviewer_probability_aggregation": shadow_config.reviewer_probability_aggregation
            }
            if shadow_config.reviewer_probability_aggregation != "vote_majority"
            else {}
        ),
        **(
            {
                "reviewer_probability_threshold": shadow_config.reviewer_probability_threshold
            }
            if shadow_config.reviewer_probability_threshold != 0.5
            else {}
        ),
        **(
            {"reviewer_probability_early_stop": True}
            if shadow_config.reviewer_probability_early_stop
            else {}
        ),
        "reviewer_system_prompt_sha256": shadow_config.reviewer_system_prompt_sha256,
        "paired_cache_source_pair_run_id": baseline_config.paired_cache_source_pair_run_id,
        "paired_cache_source_manifest_fingerprint": baseline_config.paired_cache_source_manifest_fingerprint,
        "producer_model": baseline_config.producer_model,
        "judge_model": shadow_config.judge_model,
        "user_model": baseline_config.user_model,
        "nl_evaluator_model": baseline_config.nl_evaluator_model,
        "seed": baseline_config.seed,
        "tasks_planned": planned,
        "pairs_present": len(pair_rows),
        "valid_pairs": len(valid_rows),
        "missing_task_ids": missing,
        "invalid_task_ids": [row["task_id"] for row in pair_rows if not row["valid"]],
        "score_protocol": score_protocols[0]
        if len(score_protocols) == 1
        else "mixed"
        if score_protocols
        else None,
        "score_protocols": score_protocols,
        "leaderboard_official": leaderboard_official,
        "official_tsr_label_valid": leaderboard_official,
        "baseline_protocol_tsr_valid_pairs": baseline_tsr,
        "shadow_protocol_tsr_valid_pairs": shadow_tsr,
        "tsr_delta_protocol_valid_pairs": shadow_tsr - baseline_tsr
        if baseline_tsr is not None and shadow_tsr is not None
        else None,
        "baseline_official_tsr_valid_pairs": baseline_tsr
        if leaderboard_official
        else None,
        "shadow_official_tsr_valid_pairs": shadow_tsr if leaderboard_official else None,
        "tsr_delta_official_valid_pairs": shadow_tsr - baseline_tsr
        if leaderboard_official
        and baseline_tsr is not None
        and (shadow_tsr is not None)
        else None,
        "baseline_tsr_valid_pairs": baseline_tsr,
        "shadow_tsr_valid_pairs": shadow_tsr,
        "tsr_delta_valid_pairs": shadow_tsr - baseline_tsr
        if baseline_tsr is not None and shadow_tsr is not None
        else None,
        "paired_outcomes": quadrants,
        "tasks_with_first_rejection": sum(
            (row["first_rejection_present"] for row in valid_rows)
        ),
        "paired_fail_to_success_after_rejection": sum(
            (row["paired_fail_to_success_after_rejection"] for row in valid_rows)
        ),
        "paired_success_to_failure_after_rejection": sum(
            (row["paired_success_to_failure_after_rejection"] for row in valid_rows)
        ),
        "task_pairs": pair_rows,
    }
