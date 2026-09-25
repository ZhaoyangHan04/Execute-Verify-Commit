"""Experiment-grade paired runner for the pinned STATE-Bench workflow."""

from __future__ import annotations
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import state_bench
from shadow_verifier import PROBABILITY_AGGREGATIONS, ReviewInfrastructureError
from shadow_verifier.backends import Completion
from shadow_verifier.backends.replay import ReplayCacheBackend
from shadow_verifier.experiments.artifacts import (
    atomic_json_dump,
    canonical_sha256,
    sha256_file,
)
from shadow_verifier.experiments.provenance import git_head
from shadow_verifier.policy_tests import PolicyTestSuite
from shadow_verifier.reviewers import (
    REVIEWER_PROMPT_VARIANTS,
    SemanticReviewerConfig,
    SemanticReviewerFactory,
    reviewer_prompt,
)
from state_bench.domain import get_domain_config
from state_bench.env_loader import load_task_environment
from state_bench.paths import domain_tasks_dir
from state_bench.protocol import load_default_protocol, load_split_task_ids
from state_bench.schemas import TaskDefinition
from . import PINNED_UPSTREAM_COMMIT
from .orchestrator import (
    StateBenchReviewInfrastructureError,
    StateBenchRun,
    run_state_bench_task,
)
from .participants import BackendUserSimulator, CompletionAudit, HarnessAgentFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = REPO_ROOT / "external" / "state-bench"
PAIR_SCHEMA = "state-bench-shadow-pair-v1"
REVIEWER_VARIANT = "plain_checks_v1"
_SAFE_ID = re.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_SHA256 = re.compile("[0-9a-f]{64}")
_GIT_REVISION = re.compile("[0-9a-f]{40}")
_MAX_IMPORTED_CACHE_ENTRY_BYTES = 32 * 1024 * 1024
_BASELINE_STABLE_FIELDS = (
    "task_id",
    "user_id",
    "task_summary",
    "conversation",
    "state_diff",
    "turns",
    "tool_calls",
    "tool_errors",
    "redundant_calls",
    "token_usage",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
BackendFactory = Callable[[str], Any]


@dataclass(frozen=True)
class PairBackendFactories:
    """Fresh backend constructors, separated by participant role."""

    producer: BackendFactory
    user_simulator: BackendFactory
    reviewer: BackendFactory

    def __post_init__(self) -> None:
        for name in ("producer", "user_simulator", "reviewer"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} backend factory must be callable")


@dataclass(frozen=True)
class StateBenchPairConfig:
    """Behavior-affecting settings shared by the paired arms."""

    producer_model: str
    user_model: str
    judge_model: str
    seed: int = 42
    verification_budget: int = 1
    reviewer_seed_mode: str = "fixed"
    reviewer_temperature: float = 0.0
    reviewer_soft_label_output: bool = False
    reviewer_probability_aggregation: str = "vote_majority"
    reviewer_probability_threshold: float = 0.5
    reviewer_probability_early_stop: bool = False
    max_tool_rounds: int = 16
    reviewer_variant: str = REVIEWER_VARIANT
    task_split: str = "test"
    run_idx: int = 1
    candidate_test_suite_sha256: str | None = None
    candidate_test_compiler_model: str | None = None
    include_tool_schemas_in_evidence: bool = True
    reviewer_candidate_effect_visibility: str = "full"

    def __post_init__(self) -> None:
        for name in ("producer_model", "user_model", "judge_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.seed) is not int:
            raise TypeError("seed must be int")
        if type(self.include_tool_schemas_in_evidence) is not bool:
            raise TypeError("include_tool_schemas_in_evidence must be bool")
        if self.reviewer_candidate_effect_visibility not in {"full", "hidden"}:
            raise ValueError(
                "reviewer_candidate_effect_visibility must be full or hidden"
            )
        if type(self.verification_budget) is not int:
            raise TypeError("verification_budget must be int")
        if self.verification_budget not in {1, 3, 5, 7}:
            raise ValueError("verification_budget must be one of: 1, 3, 5, 7")
        if self.reviewer_seed_mode not in {"independent", "fixed"}:
            raise ValueError("reviewer_seed_mode must be 'independent' or 'fixed'")
        if not 0.0 <= self.reviewer_temperature <= 2.0:
            raise ValueError("reviewer_temperature must be in [0, 2]")
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
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be >= 1")
        if self.task_split not in {"test", "train", "all"}:
            raise ValueError("task_split must be one of: test, train, all")
        if type(self.run_idx) is not int or self.run_idx < 1:
            raise ValueError("run_idx must be a positive integer")
        if self.reviewer_variant not in REVIEWER_PROMPT_VARIANTS:
            raise ValueError(f"unknown reviewer_variant: {self.reviewer_variant!r}")
        suite_fields = (
            self.candidate_test_suite_sha256,
            self.candidate_test_compiler_model,
        )
        if (suite_fields[0] is None) != (suite_fields[1] is None):
            raise ValueError("candidate policy-test suite identity must be all-or-none")
        if suite_fields[0] is not None:
            if (
                not isinstance(suite_fields[0], str)
                or _SHA256.fullmatch(suite_fields[0]) is None
            ):
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
        if self.reviewer_candidate_effect_visibility == "full":
            value.pop("reviewer_candidate_effect_visibility")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"schema": PAIR_SCHEMA, "config": self.artifact_config})


@dataclass(frozen=True)
class LoadedStateBenchTask:
    """Trusted-host values loaded from the pinned official checkout."""

    domain: Any
    task: TaskDefinition
    env_data: Any
    env_path: Path
    protocol: Any
    data_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class StateBenchPairResult:
    pair_id: str
    pair_dir: Path
    baseline: StateBenchRun
    shadow: StateBenchRun
    baseline_trajectory_path: Path
    shadow_trajectory_path: Path
    shadow_review_audit_path: Path
    summary_path: Path
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class _BaselineImport:
    source_projection: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _git_output(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot verify pinned STATE-Bench checkout")
    return completed.stdout.strip()


def _assert_pinned_package_import() -> None:
    package_file = Path(state_bench.__file__).resolve()
    expected = (UPSTREAM_ROOT / "state_bench" / "__init__.py").resolve()
    if package_file != expected:
        raise RuntimeError(
            "imported state_bench package is not the reviewed pinned checkout"
        )


def _assert_pinned_tracked_file(path: Path, *, subject: str) -> tuple[Path, str]:
    root = UPSTREAM_ROOT.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise RuntimeError(f"{subject} resolved outside the pinned checkout") from None
    relative_text = relative.as_posix()
    _git_output(UPSTREAM_ROOT, "ls-files", "--error-unmatch", "--", relative_text)
    working_blob = _git_output(UPSTREAM_ROOT, "hash-object", str(resolved))
    pinned_blob = _git_output(UPSTREAM_ROOT, "rev-parse", f"HEAD:{relative_text}")
    if working_blob != pinned_blob:
        raise RuntimeError(f"{subject} does not match its pinned Git blob")
    return (resolved, relative_text)


def load_official_task(
    *, domain_name: str, task_id: str, task_split: str = "test"
) -> LoadedStateBenchTask:
    """Load one active official task and its isolated task environment."""
    if not isinstance(domain_name, str) or not domain_name:
        raise ValueError("domain_name must be non-empty")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be non-empty")
    if task_split not in {"test", "train", "all"}:
        raise ValueError("task_split must be one of: test, train, all")
    head = _git_output(UPSTREAM_ROOT, "rev-parse", "HEAD")
    if head != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError("STATE-Bench checkout is not at the reviewed pin")
    if _git_output(UPSTREAM_ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("pinned STATE-Bench checkout must be clean")
    _assert_pinned_package_import()
    protocol = load_default_protocol()
    prompt_errors = protocol.validate_prompt_hashes()
    if prompt_errors:
        raise RuntimeError("official protocol prompt hashes do not match the checkout")
    if domain_name not in protocol.domains:
        raise ValueError(f"domain {domain_name!r} is outside the default protocol")
    active_ids = load_split_task_ids(domain_name, task_split, protocol.split_version)
    if task_id not in active_ids:
        raise ValueError(
            f"task {task_id!r} is not active in STATE-Bench split {task_split!r}"
        )
    domain = get_domain_config(domain_name)
    task_path = domain_tasks_dir(domain_name) / f"{task_id}.json"
    task_path, task_relative = _assert_pinned_tracked_file(
        task_path, subject="STATE-Bench task file"
    )
    task = TaskDefinition.load(task_path)
    env_data, env_path = load_task_environment(domain, task)
    env_path, env_relative = _assert_pinned_tracked_file(
        env_path, subject="STATE-Bench environment file"
    )
    split_membership = [
        split
        for split in ("train", "test")
        if task_id in load_split_task_ids(domain_name, split, protocol.split_version)
    ]
    provenance = {
        "upstream_git_head": head,
        "protocol_id": protocol.protocol_id,
        "split_version": protocol.split_version,
        "selected_split": task_split,
        "split_membership": split_membership,
        "official_protocol_split": protocol.split,
        "official_protocol_num_runs": protocol.num_runs,
        "task_file": task_relative,
        "environment_file": env_relative,
        "task_file_sha256": sha256_file(task_path),
        "environment_file_sha256": sha256_file(env_path),
        "task_sha256": canonical_sha256(task.to_dict()),
        "simulator_protocol": protocol.simulator_metadata(domain_name),
    }
    return LoadedStateBenchTask(
        domain=domain,
        task=task,
        env_data=env_data,
        env_path=env_path,
        protocol=protocol,
        data_provenance=provenance,
    )


def make_generic_reviewer_factory(
    *,
    backend_factory: BackendFactory,
    model: str,
    seed: int,
    variant: str = REVIEWER_VARIANT,
    seed_mode: str = "fixed",
    soft_label_output: bool = False,
    candidate_effect_visibility: str = "full",
) -> SemanticReviewerFactory:
    """Build V2 from generic evidence only; no task/gold parameter exists."""
    if not callable(backend_factory):
        raise TypeError("reviewer backend_factory must be callable")
    return SemanticReviewerFactory(
        backend_factory,
        model=model,
        config=SemanticReviewerConfig(
            seed=seed,
            system_prompt=reviewer_prompt(variant),
            soft_label_output=soft_label_output,
            candidate_effect_visibility=candidate_effect_visibility,
        ),
        sample_seed_mode=seed_mode,
    )


def _backend_identity(backend: Any) -> dict[str, Any]:
    model = getattr(backend, "model", None)
    config = getattr(backend, "config", None)
    if not isinstance(model, str) or not model:
        raise TypeError("completion backend.model must be non-empty")
    if is_dataclass(config) and (not isinstance(config, type)):
        config_value = asdict(config)
    elif isinstance(config, dict):
        config_value = dict(config)
    else:
        raise TypeError("completion backend.config must be a dataclass or dictionary")
    config_type = f"{type(config).__module__}.{type(config).__qualname__}"
    if config_type == "shadow_verifier.backends.dashscope.DashScopeConfig" and all(
        (
            config_value.get(key) == value
            for key, value in {
                "transport_max_attempts": 1,
                "transport_min_interval_seconds": 0.0,
                "transport_retry_backoff_seconds": 30.0,
            }.items()
        )
    ):
        for key in (
            "transport_max_attempts",
            "transport_min_interval_seconds",
            "transport_retry_backoff_seconds",
        ):
            config_value.pop(key)
    config_sha256 = canonical_sha256({"type": config_type, "value": config_value})
    safe_config = {
        name: config_value[name]
        for name in (
            "temperature",
            "top_p",
            "max_completion_tokens",
            "enable_thinking",
            "timeout_seconds",
            "reasoning_effort",
            "base_url",
            "protocol",
            "max_rationale_chars",
            "probability_mass_tolerance",
        )
        if name in config_value
    }
    return {
        "implementation": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "model": model,
        "config_type": config_type,
        "config_sha256": config_sha256,
        "generation_config": safe_config,
    }


class _AuditedCompletionBackend:
    """Record digest-only Judge telemetry without persisting its evidence."""

    def __init__(self, backend: Any, *, participant: str) -> None:
        identity = _backend_identity(backend)
        self._backend = backend
        self.model = identity["model"]
        self.config = backend.config
        self.participant = participant
        self.completion_audits: list[CompletionAudit] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        request_sha256 = canonical_sha256(
            {
                "backend": _backend_identity(self._backend),
                "messages": messages,
                "tools": tools,
                "seed": seed,
                "json_mode": json_mode,
            }
        )
        completion = self._backend.complete(
            messages=messages, tools=tools, seed=seed, json_mode=json_mode
        )
        if not isinstance(completion, Completion):
            raise TypeError("reviewer backend must return Completion")
        completion_sha256 = canonical_sha256(asdict(completion))
        self.completion_audits.append(
            CompletionAudit(
                participant=self.participant,
                ordinal=len(self.completion_audits) + 1,
                model=completion.model,
                request_sha256=request_sha256,
                completion_sha256=completion_sha256,
                cache_hit=bool(getattr(self._backend, "last_cache_hit", False)),
                latency_s=float(completion.latency_s),
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )
        )
        return completion


def _task_slug(domain_name: str, task_id: str) -> str:
    digest = hashlib.sha256(f"{domain_name}\x00{task_id}".encode()).hexdigest()[:24]
    return f"{domain_name}-{digest}"


def _random_safe_id(prefix: str, *, forbidden: str = "") -> str:
    for _ in range(16):
        candidate = f"{prefix}-{secrets.token_hex(16)}"
        if _SAFE_ID.fullmatch(candidate) and (
            not forbidden or forbidden not in candidate
        ):
            return candidate
    raise RuntimeError("could not generate an opaque experiment identifier")


def _load_artifact_object(path: Path, *, subject: str) -> dict[str, Any]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{subject} must be a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {subject}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{subject} must contain a JSON object")
    return value


def _baseline_projection(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    return {field: trajectory.get(field) for field in _BASELINE_STABLE_FIELDS}


def _require_real_directory(path: Path, *, subject: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{subject} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{subject} must be a real directory")


def _copy_cache_entry(
    source: Path, destination: Path, *, expected_sha256: str | None
) -> None:
    try:
        info = source.lstat()
    except OSError as exc:
        raise ValueError("baseline source cache entry is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("baseline source cache entry must be a regular file")
    if info.st_size > _MAX_IMPORTED_CACHE_ENTRY_BYTES:
        raise ValueError("baseline source cache entry is unreasonably large")
    if expected_sha256 is not None:
        if not _SHA256.fullmatch(expected_sha256):
            raise ValueError("baseline source completion digest is invalid")
        if sha256_file(source) != expected_sha256:
            raise ValueError("baseline source cache entry digest mismatch")
    destination.parent.mkdir(mode=448, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 384)
        with (
            source.open("rb") as input_handle,
            os.fdopen(descriptor, "wb") as output_handle,
        ):
            descriptor = -1
            for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                output_handle.write(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cache_rows(summary: Mapping[str, Any], *, role: str) -> dict[str, str]:
    baseline = summary.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("baseline source summary is missing baseline metadata")
    field = (
        "producer_completions" if role == "producer" else "user_simulator_completions"
    )
    rows = baseline.get(field)
    if not isinstance(rows, list):
        raise ValueError(f"baseline source summary is missing {field}")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("baseline source completion audit is malformed")
        request_sha256 = row.get("request_sha256")
        completion_sha256 = row.get("completion_sha256")
        if not isinstance(request_sha256, str) or not _SHA256.fullmatch(request_sha256):
            raise ValueError("baseline source request digest is invalid")
        if not isinstance(completion_sha256, str) or not _SHA256.fullmatch(
            completion_sha256
        ):
            raise ValueError("baseline source completion digest is invalid")
        previous = result.setdefault(request_sha256, completion_sha256)
        if previous != completion_sha256:
            raise ValueError("one baseline request maps to multiple completions")
    return result


def _prepare_baseline_import(
    *,
    source_pair_dir: Path,
    destination_cache_root: Path,
    loaded: LoadedStateBenchTask,
    config: StateBenchPairConfig,
    producer_identity: Mapping[str, Any],
    user_identity: Mapping[str, Any],
) -> _BaselineImport:
    if not isinstance(source_pair_dir, Path):
        raise TypeError("baseline_source_pair_dir must be a pathlib.Path")
    _require_real_directory(source_pair_dir, subject="baseline source pair path")
    source_pair_dir = source_pair_dir.resolve()
    expected_task_root = _task_slug(loaded.domain.name, str(loaded.task.task_id))
    if source_pair_dir.parent.name != expected_task_root:
        raise ValueError("baseline source pair belongs to a different domain or task")
    status_path = source_pair_dir / "status.json"
    status = _load_artifact_object(status_path, subject="baseline source status")
    source_status = status.get("status")
    if source_status not in {"complete", "failed"}:
        raise ValueError("baseline source pair must have terminal status")
    source_pair_id = status.get("pair_id")
    source_fingerprint = status.get("pair_common_fingerprint")
    if not isinstance(source_pair_id, str) or not _SAFE_ID.fullmatch(source_pair_id):
        raise ValueError("baseline source pair_id is invalid")
    if not isinstance(source_fingerprint, str) or not _SHA256.fullmatch(
        source_fingerprint
    ):
        raise ValueError("baseline source pair fingerprint is invalid")
    trajectory_path = source_pair_dir / "baseline" / "trajectory.json"
    trajectory = _load_artifact_object(
        trajectory_path, subject="baseline source trajectory"
    )
    if trajectory.get("task_id") != loaded.task.task_id:
        raise ValueError("baseline source trajectory task_id mismatch")
    agent_model = trajectory.get("agent_model")
    actual_simulator = trajectory.get("actual_simulator")
    task_selection = trajectory.get("task_selection")
    if (
        not isinstance(agent_model, Mapping)
        or agent_model.get("model_name") != config.producer_model
    ):
        raise ValueError("baseline source Producer model mismatch")
    if (
        not isinstance(actual_simulator, Mapping)
        or actual_simulator.get("model") != config.user_model
    ):
        raise ValueError("baseline source UserSimulator model mismatch")
    if actual_simulator.get("backend") != user_identity:
        raise ValueError("baseline source UserSimulator backend mismatch")
    if (
        not isinstance(task_selection, Mapping)
        or task_selection.get("split") != config.task_split
    ):
        raise ValueError("baseline source split mismatch")
    if task_selection.get("run_idx") != config.run_idx:
        raise ValueError("baseline source run_idx mismatch")
    summary: dict[str, Any] | None = None
    summary_path = source_pair_dir / "pair_summary.json"
    selected_entries: dict[str, dict[str, str | None]] = {}
    if source_status == "complete":
        summary = _load_artifact_object(summary_path, subject="baseline source summary")
        pair_identity = summary.get("pair_identity")
        if not isinstance(pair_identity, Mapping) or (
            pair_identity.get("pair_id") != source_pair_id
            or pair_identity.get("domain") != loaded.domain.name
            or pair_identity.get("task_id") != loaded.task.task_id
            or (pair_identity.get("pair_common_fingerprint") != source_fingerprint)
        ):
            raise ValueError("baseline source summary identity mismatch")
        source_config = summary.get("config")
        if not isinstance(source_config, Mapping):
            raise ValueError("baseline source summary config is missing")
        for field in (
            "producer_model",
            "user_model",
            "seed",
            "max_tool_rounds",
            "task_split",
            "run_idx",
        ):
            if source_config.get(field) != getattr(config, field):
                raise ValueError(f"baseline source config mismatch: {field}")
        if summary.get("data_provenance") != dict(loaded.data_provenance):
            raise ValueError("baseline source data provenance mismatch")
        backends = summary.get("backend_provenance")
        if not isinstance(backends, Mapping):
            raise ValueError("baseline source backend provenance is missing")
        if (
            backends.get("producer") != producer_identity
            or backends.get("user_simulator") != user_identity
        ):
            raise ValueError("baseline source participant backend mismatch")
        baseline_summary = summary.get("baseline")
        if not isinstance(baseline_summary, Mapping) or baseline_summary.get(
            "trajectory_sha256"
        ) != canonical_sha256(trajectory):
            raise ValueError("baseline source trajectory digest mismatch")
        for role in ("producer", "user_simulator"):
            selected_entries[role] = {
                request: completion
                for request, completion in _cache_rows(summary, role=role).items()
            }
    else:
        for role in ("producer", "user_simulator"):
            role_dir = source_pair_dir / "_completion_cache" / source_fingerprint / role
            _require_real_directory(
                role_dir, subject="failed baseline source replay cache"
            )
            entries: dict[str, str | None] = {}
            for path in sorted(role_dir.iterdir()):
                if path.suffix != ".json" or not _SHA256.fullmatch(path.stem):
                    raise ValueError(
                        "failed baseline source cache has an unexpected entry"
                    )
                entries[path.stem] = None
            selected_entries[role] = entries
    source_cache_root = source_pair_dir / "_completion_cache" / source_fingerprint
    _require_real_directory(source_cache_root, subject="baseline source replay cache")
    imported_counts: dict[str, int] = {}
    for role, entries in selected_entries.items():
        _require_real_directory(
            source_cache_root / role, subject=f"baseline source {role} replay cache"
        )
        imported_counts[role] = len(entries)
        for request_sha256, completion_sha256 in entries.items():
            _copy_cache_entry(
                source_cache_root / role / f"{request_sha256}.json",
                destination_cache_root / role / f"{request_sha256}.json",
                expected_sha256=completion_sha256,
            )
    projection = _baseline_projection(trajectory)
    provenance = {
        "source_pair_id": source_pair_id,
        "source_pair_status": source_status,
        "source_status_sha256": sha256_file(status_path),
        "source_summary_sha256": sha256_file(summary_path)
        if summary is not None
        else None,
        "source_baseline_trajectory_sha256": sha256_file(trajectory_path),
        "source_baseline_semantic_sha256": canonical_sha256(projection),
        "source_pair_common_fingerprint": source_fingerprint,
        "imported_cache_entries": imported_counts,
    }
    return _BaselineImport(source_projection=projection, provenance=provenance)


def _implementation_provenance() -> dict[str, Any]:
    workflow_dir = Path(__file__).parent
    shadow_dir = REPO_ROOT / "shadow-verifier" / "src" / "shadow_verifier"
    paths = sorted(workflow_dir.glob("*.py")) + [
        shadow_dir / "model.py",
        shadow_dir / "protocols.py",
        shadow_dir / "runtime.py",
        shadow_dir / "backends" / "dashscope.py",
        shadow_dir / "backends" / "replay.py",
        shadow_dir / "reviewers" / "semantic.py",
        shadow_dir / "reviewers" / "prompts.py",
    ]
    files = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}
    return {
        "implementation_git_head": git_head(REPO_ROOT),
        "files": files,
        "fingerprint": canonical_sha256(files),
    }


def _audit_rows(values: tuple[CompletionAudit, ...]) -> list[dict[str, Any]]:
    return [value.to_dict() for value in values]


def _arm_summary(
    run: StateBenchRun,
    agent_factory: HarnessAgentFactory,
    simulator: BackendUserSimulator,
) -> dict[str, Any]:
    if agent_factory.agent is None:
        raise RuntimeError("STATE-Bench did not construct its producer")
    producer = agent_factory.agent.completion_audits
    user = simulator.completion_audits
    return {
        "mode": run.mode,
        "trajectory_sha256": canonical_sha256(run.trajectory.to_dict()),
        "public_history_sha256": canonical_sha256(list(run.public_history)),
        "final_snapshot_sha256": canonical_sha256(run.final_snapshot),
        "execution_metrics": dict(run.execution_metrics),
        "producer_completions": _audit_rows(producer),
        "user_simulator_completions": _audit_rows(user),
        "physical_api_calls": sum((not row.cache_hit for row in (*producer, *user))),
        "replay_cache_hits": sum((row.cache_hit for row in (*producer, *user))),
    }


def _validate_pre_rejection_replay(
    *,
    baseline_agent_factory: HarnessAgentFactory,
    baseline_simulator: BackendUserSimulator,
    shadow_agent_factory: HarnessAgentFactory,
    shadow_simulator: BackendUserSimulator,
    release_marker: Mapping[str, int] | None,
) -> None:
    if baseline_agent_factory.agent is None or shadow_agent_factory.agent is None:
        raise RuntimeError("shadow producer was not constructed")
    baseline_producer = baseline_agent_factory.agent.completion_audits
    baseline_user = baseline_simulator.completion_audits
    shadow_producer = shadow_agent_factory.agent.completion_audits
    shadow_user = shadow_simulator.completion_audits
    producer_limit = (
        release_marker["producer_completions"]
        if release_marker is not None
        else len(shadow_producer)
    )
    user_limit = (
        release_marker["user_simulator_completions"]
        if release_marker is not None
        else len(shadow_user)
    )
    for subject, baseline, shadow, limit in (
        ("producer", baseline_producer, shadow_producer, producer_limit),
        ("user simulator", baseline_user, shadow_user, user_limit),
    ):
        if len(baseline) < limit or len(shadow) < limit:
            raise RuntimeError(
                f"{subject} replay prefix is shorter than its release marker"
            )
        if release_marker is None and len(baseline) != len(shadow):
            raise RuntimeError(
                f"{subject} completion counts differ without a rejection"
            )
        for index, (recorded, replayed) in enumerate(
            zip(baseline[:limit], shadow[:limit], strict=True), 1
        ):
            if not replayed.cache_hit:
                raise RuntimeError(
                    f"paired {subject} performed a physical call before first rejection"
                )
            if (
                recorded.request_sha256 is None
                or recorded.completion_sha256 is None
                or replayed.request_sha256 is None
                or (replayed.completion_sha256 is None)
            ):
                raise RuntimeError(
                    f"paired {subject} audit is missing a completion digest"
                )
            if (
                recorded.request_sha256 != replayed.request_sha256
                or recorded.completion_sha256 != replayed.completion_sha256
            ):
                raise RuntimeError(
                    f"paired {subject} completion {index} is not the baseline prefix"
                )


def run_pair(
    *,
    loaded: LoadedStateBenchTask,
    config: StateBenchPairConfig,
    backend_factories: PairBackendFactories,
    output_dir: Path,
    baseline_source_pair_dir: Path | None = None,
    candidate_test_suite: PolicyTestSuite | None = None,
    first_rejection_diagnostic: Any | None = None,
    continuation_cache_root: Path | None = None,
) -> StateBenchPairResult:
    """Record baseline completions, then replay them in one Shadow arm.

    Shadow's Producer and UserSimulator are replay-only until the first actual
    rejection.  Any earlier cache miss raises ``ReplayCacheMissError`` before
    reaching a physical backend.  The rejection callback then irreversibly
    releases both participants so the repaired trajectory can diverge.
    """
    if first_rejection_diagnostic is not None and (
        baseline_source_pair_dir is None or config.verification_budget != 1
    ):
        raise ValueError(
            "first-rejection diagnostic requires historical baseline and B1"
        )
    if not isinstance(loaded, LoadedStateBenchTask):
        raise TypeError("loaded must come from load_official_task()")
    if not isinstance(config, StateBenchPairConfig):
        raise TypeError("config must be StateBenchPairConfig")
    if not isinstance(backend_factories, PairBackendFactories):
        raise TypeError("backend_factories must be PairBackendFactories")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    domain_name = getattr(loaded.domain, "name", None)
    semantic_task_id = getattr(loaded.task, "task_id", None)
    if not isinstance(domain_name, str) or not isinstance(semantic_task_id, str):
        raise TypeError("loaded task is missing its official domain/task identity")
    if candidate_test_suite is None:
        if config.candidate_test_suite_sha256 is not None:
            raise ValueError(
                "config enables candidate policy tests but no suite was supplied"
            )
    else:
        if not isinstance(candidate_test_suite, PolicyTestSuite):
            raise TypeError("candidate_test_suite must be PolicyTestSuite or None")
        if (
            candidate_test_suite.benchmark != "state_bench"
            or candidate_test_suite.domain != domain_name
            or candidate_test_suite.compiler_model != config.producer_model
            or (candidate_test_suite.suite_sha256 != config.candidate_test_suite_sha256)
        ):
            raise ValueError("candidate policy-test suite identity mismatch")
    reloaded = load_official_task(
        domain_name=domain_name, task_id=semantic_task_id, task_split=config.task_split
    )
    if dict(loaded.data_provenance) != dict(reloaded.data_provenance):
        raise ValueError("loaded task provenance does not match the pinned checkout")
    loaded = reloaded
    task = loaded.task
    domain = loaded.domain
    if (
        config.task_split == loaded.protocol.split
        and config.run_idx > loaded.protocol.num_runs
    ):
        raise ValueError(
            f"run_idx must be within 1..{loaded.protocol.num_runs} for the official {loaded.protocol.split!r} protocol split"
        )
    pair_id = _random_safe_id("state-pair")
    episode_id = _random_safe_id("state-episode", forbidden=str(task.task_id))
    pair_dir = output_dir / _task_slug(domain.name, str(task.task_id)) / pair_id
    if pair_dir.exists():
        raise FileExistsError("random pair artifact directory already exists")
    pair_dir.mkdir(parents=True, exist_ok=False)
    status_path = pair_dir / "status.json"
    review_audit_path = pair_dir / "shadow" / "review_audit.json"
    incremental_review_events: list[dict[str, Any]] = []

    def persist_review_event(event: Mapping[str, Any]) -> None:
        cloned = json.loads(
            json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
        incremental_review_events.append(cloned)
        atomic_json_dump(
            review_audit_path,
            {
                "schema": "state-bench-shadow-review-audit-v1",
                "status": "running",
                "pair_id": pair_id,
                "record_count": len(incremental_review_events),
                "records": incremental_review_events,
            },
        )

    baseline_producer_raw = backend_factories.producer(config.producer_model)
    shadow_producer_raw = backend_factories.producer(config.producer_model)
    baseline_user_raw = backend_factories.user_simulator(config.user_model)
    shadow_user_raw = backend_factories.user_simulator(config.user_model)
    reviewer_runtime_model = config.judge_model
    assert reviewer_runtime_model is not None
    reviewer_raw = backend_factories.reviewer(reviewer_runtime_model)
    if (
        baseline_producer_raw is shadow_producer_raw
        or baseline_user_raw is shadow_user_raw
    ):
        raise ValueError("each pair arm requires fresh participant backend instances")
    producer_identity = _backend_identity(baseline_producer_raw)
    user_identity = _backend_identity(baseline_user_raw)
    if producer_identity != _backend_identity(shadow_producer_raw):
        raise ValueError("producer backend configuration differs across pair arms")
    if user_identity != _backend_identity(shadow_user_raw):
        raise ValueError(
            "user simulator backend configuration differs across pair arms"
        )
    reviewer_identity = _backend_identity(reviewer_raw)
    if continuation_cache_root is not None:
        from workflow.experiment_replay import OccurrenceCacheBackend

        shadow_producer_raw = OccurrenceCacheBackend(
            shadow_producer_raw, continuation_cache_root / "producer"
        )
        shadow_user_raw = OccurrenceCacheBackend(
            shadow_user_raw, continuation_cache_root / "user"
        )
        reviewer_raw = OccurrenceCacheBackend(
            reviewer_raw, continuation_cache_root / "reviewer"
        )
    reviewer_backend = _AuditedCompletionBackend(reviewer_raw, participant="reviewer")

    def reviewer_backend_factory(model: str) -> _AuditedCompletionBackend:
        if model != reviewer_runtime_model:
            raise ValueError("reviewer factory received an unexpected model")
        return reviewer_backend

    simulator_prompt = domain.build_simulator_prompt(
        task, loaded.env_data, task.user_id
    )
    simulator_prompt_sha256 = hashlib.sha256(simulator_prompt.encode()).hexdigest()
    implementation = _implementation_provenance()
    pair_common = {
        "schema": PAIR_SCHEMA,
        "config_fingerprint": config.fingerprint,
        "data_fingerprint": canonical_sha256(dict(loaded.data_provenance)),
        "implementation_fingerprint": implementation["fingerprint"],
        "producer_backend": producer_identity,
        "user_simulator_backend": user_identity,
        "reviewer_backend": reviewer_identity,
        "simulator_prompt_sha256": simulator_prompt_sha256,
    }
    pair_common_fingerprint = canonical_sha256(pair_common)
    cache_root = pair_dir / "_completion_cache" / pair_common_fingerprint
    baseline_import = (
        _prepare_baseline_import(
            source_pair_dir=baseline_source_pair_dir,
            destination_cache_root=cache_root,
            loaded=loaded,
            config=config,
            producer_identity=producer_identity,
            user_identity=user_identity,
        )
        if baseline_source_pair_dir is not None
        else None
    )
    baseline_producer = ReplayCacheBackend(
        baseline_producer_raw,
        cache_dir=cache_root / "producer",
        replay_only=baseline_import is not None,
    )
    baseline_user = ReplayCacheBackend(
        baseline_user_raw,
        cache_dir=cache_root / "user_simulator",
        replay_only=baseline_import is not None,
    )
    shadow_producer = ReplayCacheBackend(
        shadow_producer_raw, cache_dir=cache_root / "producer", replay_only=True
    )
    shadow_user = ReplayCacheBackend(
        shadow_user_raw, cache_dir=cache_root / "user_simulator", replay_only=True
    )
    official_simulator_requirement = loaded.protocol.simulator_metadata(domain.name)
    official_protocol_requirement = {
        "protocol_id": loaded.protocol.protocol_id,
        "split": loaded.protocol.split,
        "num_runs": loaded.protocol.num_runs,
    }
    common_metadata = {
        "workflow": PAIR_SCHEMA,
        "pair_id": pair_id,
        "pair_common_fingerprint": pair_common_fingerprint,
        "config_fingerprint": config.fingerprint,
        "agent_model": {"model_name": config.producer_model, "reasoning_level": None},
        "leaderboard_official": False,
        "protocol_compatibility": "custom_user_simulator_nonofficial",
        "task_selection": {"split": config.task_split, "run_idx": config.run_idx},
        "actual_simulator": {
            "model": config.user_model,
            "backend": user_identity,
            "assembled_prompt_sha256": simulator_prompt_sha256,
        },
        "official_simulator_requirement": official_simulator_requirement,
        "official_protocol_requirement": official_protocol_requirement,
    }
    atomic_json_dump(
        status_path,
        {
            "status": "running",
            "pair_id": pair_id,
            "pair_common_fingerprint": pair_common_fingerprint,
        },
    )
    baseline_agent_factory = HarnessAgentFactory(
        backend=baseline_producer, seed=config.seed
    )
    baseline_simulator = BackendUserSimulator(
        backend=baseline_user, system_prompt=simulator_prompt, seed=config.seed
    )
    try:
        baseline = run_state_bench_task(
            task=task,
            env_data=loaded.env_data,
            user_id=task.user_id,
            domain=domain,
            agent_factory=baseline_agent_factory,
            simulator=baseline_simulator,
            mode="baseline",
            agent_episode_id=episode_id,
            max_tool_rounds=config.max_tool_rounds,
            trajectory_metadata={
                **common_metadata,
                "pair_arm": "baseline_imported_replay"
                if baseline_import is not None
                else "baseline_record",
            },
        )
        if baseline_import is not None:
            if _baseline_projection(baseline.trajectory.to_dict()) != dict(
                baseline_import.source_projection
            ):
                raise RuntimeError("imported baseline semantic trajectory mismatch")
            if baseline_agent_factory.agent is None:
                raise RuntimeError("imported baseline did not construct its Producer")
            imported_audits = (
                *baseline_agent_factory.agent.completion_audits,
                *baseline_simulator.completion_audits,
            )
            if any((not row.cache_hit for row in imported_audits)):
                raise RuntimeError("imported baseline performed a physical API call")
        baseline_path = pair_dir / "baseline" / "trajectory.json"
        atomic_json_dump(baseline_path, baseline.trajectory.to_dict())
        shadow_agent_factory = HarnessAgentFactory(
            backend=shadow_producer, seed=config.seed
        )
        shadow_simulator = BackendUserSimulator(
            backend=shadow_user, system_prompt=simulator_prompt, seed=config.seed
        )
        release_marker: dict[str, int] | None = None

        def release_after_first_rejection() -> None:
            nonlocal release_marker
            if release_marker is not None:
                raise RuntimeError(
                    "first-rejection release callback ran more than once"
                )
            agent = shadow_agent_factory.agent
            if agent is None:
                raise RuntimeError("rejection arrived before producer construction")
            release_marker = {
                "producer_completions": len(agent.completion_audits),
                "user_simulator_completions": len(shadow_simulator.completion_audits),
            }
            shadow_producer.release_replay_only()
            shadow_user.release_replay_only()

        shadow = run_state_bench_task(
            task=task,
            env_data=loaded.env_data,
            user_id=task.user_id,
            domain=domain,
            agent_factory=shadow_agent_factory,
            simulator=shadow_simulator,
            mode="shadow",
            reviewer_factory=first_rejection_diagnostic.factory
            if first_rejection_diagnostic is not None
            else make_generic_reviewer_factory(
                backend_factory=reviewer_backend_factory,
                model=reviewer_runtime_model,
                seed=config.seed,
                variant=config.reviewer_variant,
                seed_mode=config.reviewer_seed_mode,
                soft_label_output=config.reviewer_soft_label_output,
                candidate_effect_visibility=config.reviewer_candidate_effect_visibility,
            ),
            verification_budget=config.verification_budget,
            probability_aggregation=config.reviewer_probability_aggregation,
            probability_threshold=config.reviewer_probability_threshold,
            probability_early_stop=config.reviewer_probability_early_stop,
            include_tool_schemas_in_evidence=config.include_tool_schemas_in_evidence,
            agent_episode_id=episode_id,
            max_tool_rounds=config.max_tool_rounds,
            trajectory_metadata={**common_metadata, "pair_arm": "shadow_replay"},
            on_first_rejection=release_after_first_rejection,
            first_rejection_diagnostic=first_rejection_diagnostic,
            on_review_event=persist_review_event,
            candidate_test_suite=candidate_test_suite,
        )
        reviewed = int(shadow.execution_metrics.get("reviewed_batches", -1))
        if reviewed != len(shadow.audit_records) or reviewed != len(
            shadow.review_events
        ):
            raise RuntimeError(
                "Shadow review metrics and audit records do not reconcile"
            )
        review_votes = sum(
            (len(record.review_decisions) for record in shadow.audit_records)
        )
        expected_review_attempts = (
            review_votes
            if config.reviewer_probability_early_stop
            else reviewed * config.verification_budget
        )
        if first_rejection_diagnostic is not None:
            first_rejection_diagnostic.assert_complete()
            expected_review_attempts = 0
        if expected_review_attempts != len(reviewer_backend.completion_audits):
            raise RuntimeError(
                "Shadow review records and reviewer attempts do not reconcile"
            )
        if list(shadow.review_events) != incremental_review_events:
            raise RuntimeError("incremental and in-memory Shadow review audits differ")
        rejected_from_records = sum(
            (not record.decision.accept for record in shadow.audit_records)
        )
        if rejected_from_records != int(
            shadow.execution_metrics.get("rejected_batches", -1)
        ):
            raise RuntimeError(
                "Shadow rejection metrics and decisions do not reconcile"
            )
        _validate_pre_rejection_replay(
            baseline_agent_factory=baseline_agent_factory,
            baseline_simulator=baseline_simulator,
            shadow_agent_factory=shadow_agent_factory,
            shadow_simulator=shadow_simulator,
            release_marker=release_marker,
        )
        rejected = int(shadow.execution_metrics.get("rejected_batches", 0))
        if bool(rejected) != (release_marker is not None):
            raise RuntimeError("replay release does not match Shadow rejection audit")
        if release_marker is None and (
            baseline.public_history != shadow.public_history
            or baseline.final_snapshot != shadow.final_snapshot
        ):
            raise RuntimeError(
                "baseline and Shadow canonical state diverged without a rejection"
            )
        shadow_path = pair_dir / "shadow" / "trajectory.json"
        atomic_json_dump(shadow_path, shadow.trajectory.to_dict())
        atomic_json_dump(
            review_audit_path,
            {
                "schema": "state-bench-shadow-review-audit-v1",
                "status": "complete",
                "pair_id": pair_id,
                "record_count": len(shadow.review_events),
                "records": list(shadow.review_events),
            },
        )
        baseline_summary = _arm_summary(
            baseline, baseline_agent_factory, baseline_simulator
        )
        shadow_summary = _arm_summary(shadow, shadow_agent_factory, shadow_simulator)
        shadow_summary["reviewer_completions"] = _audit_rows(
            tuple(reviewer_backend.completion_audits)
        )
        reviewer_logical_api_calls = sum(
            (not row.cache_hit for row in reviewer_backend.completion_audits)
        )
        shadow_summary["reviewer_physical_api_calls"] = reviewer_logical_api_calls
        shadow_summary["physical_api_calls_including_reviewer"] = (
            shadow_summary["physical_api_calls"]
            + shadow_summary["reviewer_physical_api_calls"]
        )
        summary = {
            "schema": PAIR_SCHEMA,
            "status": "complete",
            "pair_identity": {
                "pair_id": pair_id,
                "domain": domain.name,
                "task_id": task.task_id,
                "episode_id_sha256": hashlib.sha256(episode_id.encode()).hexdigest(),
                "config_fingerprint": config.fingerprint,
                "pair_common_fingerprint": pair_common_fingerprint,
            },
            "config": config.artifact_config,
            "data_provenance": dict(loaded.data_provenance),
            "implementation_provenance": implementation,
            "backend_provenance": {
                "producer": producer_identity,
                "user_simulator": user_identity,
                "reviewer": {
                    **reviewer_identity,
                    "variant": config.reviewer_variant,
                    "seed_mode": config.reviewer_seed_mode,
                    **(
                        {"soft_label_output": True}
                        if config.reviewer_soft_label_output
                        else {}
                    ),
                    **{},
                    "system_prompt_sha256": hashlib.sha256(
                        reviewer_prompt(config.reviewer_variant).encode()
                    ).hexdigest(),
                },
            },
            **{},
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
            "pairing": {
                "baseline_mode": "imported_exact_replay"
                if baseline_import is not None
                else "record",
                "baseline_source": dict(baseline_import.provenance)
                if baseline_import is not None
                else None,
                "baseline_replay_validated": baseline_import is not None,
                "shadow_mode": "replay_until_first_rejection",
                "released_after_first_rejection": release_marker is not None,
                "release_marker": release_marker,
            },
            "artifacts": {
                "baseline_trajectory": str(baseline_path.relative_to(pair_dir)),
                "shadow_trajectory": str(shadow_path.relative_to(pair_dir)),
                "shadow_review_audit": str(review_audit_path.relative_to(pair_dir)),
                "shadow_review_audit_sha256": sha256_file(review_audit_path),
            },
            "baseline": baseline_summary,
            "shadow": shadow_summary,
            "leaderboard_official": False,
            "official_simulator_requirement": official_simulator_requirement,
            "official_protocol_requirement": official_protocol_requirement,
            "official_scoring": "not_run",
        }
        summary_path = pair_dir / "pair_summary.json"
        atomic_json_dump(summary_path, summary)
        atomic_json_dump(
            status_path,
            {
                "status": "complete",
                "pair_id": pair_id,
                "pair_common_fingerprint": pair_common_fingerprint,
                "summary_sha256": sha256_file(summary_path),
            },
        )
        return StateBenchPairResult(
            pair_id=pair_id,
            pair_dir=pair_dir,
            baseline=baseline,
            shadow=shadow,
            baseline_trajectory_path=baseline_path,
            shadow_trajectory_path=shadow_path,
            shadow_review_audit_path=review_audit_path,
            summary_path=summary_path,
            summary=summary,
        )
    except Exception as exc:
        failure_fields: dict[str, Any] = {}
        artifact_errors: list[dict[str, str]] = []
        if isinstance(exc, ReviewInfrastructureError):
            failure_fields["failure_classification"] = "review_infrastructure"
        if isinstance(exc, StateBenchReviewInfrastructureError):
            expected_events = list(exc.review_events)
            if not incremental_review_events:
                incremental_review_events.extend(expected_events)
            elif incremental_review_events != expected_events:
                failure_fields["review_audit_consistency"] = "mismatch"
            failure_path = pair_dir / "shadow" / "review_failure.json"
            decision = exc.review_event.get("decision", {})
            if not isinstance(decision, Mapping):
                decision = {}
            review_failure_fields = {
                "code": decision.get("code"),
                "outcome": exc.review_event.get("outcome"),
                "evidence_sha256": exc.review_event.get("evidence_sha256"),
            }
            try:
                atomic_json_dump(
                    failure_path,
                    {
                        "schema": "state-bench-shadow-review-failure-v1",
                        "pair_id": pair_id,
                        "record_count": len(exc.review_events),
                        "records": list(exc.review_events),
                        "review": exc.review_event,
                    },
                )
                review_failure_fields.update(
                    {
                        "artifact": str(failure_path.relative_to(pair_dir)),
                        "artifact_sha256": sha256_file(failure_path),
                    }
                )
            except Exception as artifact_exc:
                artifact_errors.append(
                    {
                        "artifact": "review_failure",
                        "error_type": type(artifact_exc).__name__,
                    }
                )
            failure_fields["review_failure"] = review_failure_fields
        try:
            atomic_json_dump(
                review_audit_path,
                {
                    "schema": "state-bench-shadow-review-audit-v1",
                    "status": "failed",
                    "pair_id": pair_id,
                    "record_count": len(incremental_review_events),
                    "records": incremental_review_events,
                    "error_type": type(exc).__name__,
                },
            )
            failure_fields["shadow_review_audit"] = {
                "artifact": str(review_audit_path.relative_to(pair_dir)),
                "artifact_sha256": sha256_file(review_audit_path),
                "record_count": len(incremental_review_events),
            }
        except Exception as artifact_exc:
            artifact_errors.append(
                {
                    "artifact": "shadow_review_audit",
                    "error_type": type(artifact_exc).__name__,
                }
            )
        if artifact_errors:
            failure_fields["failure_artifact_errors"] = artifact_errors
        provider_exception_type = getattr(exc, "provider_exception_type", None)
        if isinstance(provider_exception_type, str) and provider_exception_type:
            failure_fields["provider_exception_type"] = provider_exception_type
        try:
            atomic_json_dump(
                status_path,
                {
                    "status": "failed",
                    "pair_id": pair_id,
                    "pair_common_fingerprint": pair_common_fingerprint,
                    "error_type": type(exc).__name__,
                    **failure_fields,
                },
            )
        except Exception:
            pass
        raise
