"""Reproducible baseline and Shadow runs on the original τ-bench test split."""

from __future__ import annotations
import copy
import functools
import hashlib
import importlib.util
import json
import platform
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
from shadow_verifier import PROBABILITY_AGGREGATIONS, Published, ShadowVerifier
from shadow_verifier.backends.dashscope import DashScopeBackend, DashScopeConfig
from shadow_verifier.backends.hosted import hosted_backend
from shadow_verifier.backends.replay import ReplayCacheBackend
from shadow_verifier.experiments import (
    atomic_json_dump,
    canonical_sha256,
    git_head,
    require_clean_git_tree,
    sha256_file,
)
from shadow_verifier.reviewers import (
    MAX_REVIEW_COMPLETION_TOKENS,
    SemanticReviewerConfig,
    SemanticReviewerFactory,
    reviewer_prompt,
)
from .environment import TauBenchForkExecutor
from .evaluation import evaluate
from .model import TauBenchToolAction
from .participants import BackendUserSimulator, ParticipantMetrics

Domain = Literal["retail", "airline"]
Mode = Literal["baseline", "shadow"]
PairingMode = Literal["independent", "record", "replay"]
SUPPORTED_DOMAINS: tuple[Domain, ...] = ("retail", "airline")
TASK_COUNTS: dict[Domain, int] = {"retail": 115, "airline": 50}
REPO_ROOT = Path(__file__).resolve().parents[2]
TAU_BENCH_ROOT = REPO_ROOT / "external" / "tau-bench"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "tau_bench"
RUNNER_SCHEMA = "shadow-tau-bench-legacy-test-v1"
MUTATING_TOOLS: dict[Domain, frozenset[str]] = {
    "retail": frozenset(
        {
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
        }
    ),
    "airline": frozenset(
        {
            "book_reservation",
            "cancel_reservation",
            "send_certificate",
            "update_reservation_baggages",
            "update_reservation_flights",
            "update_reservation_passengers",
        }
    ),
}
READ_ONLY_TOOLS: dict[Domain, frozenset[str]] = {
    "retail": frozenset(
        {
            "calculate",
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "get_order_details",
            "get_product_details",
            "get_user_details",
            "list_all_product_types",
            "think",
            "transfer_to_human_agents",
        }
    ),
    "airline": frozenset(
        {
            "calculate",
            "get_reservation_details",
            "get_user_details",
            "list_all_airports",
            "search_direct_flight",
            "search_onestop_flight",
            "think",
            "transfer_to_human_agents",
        }
    ),
}


@dataclass(frozen=True)
class TauBenchConfig:
    producer_model: str
    judge_model: str
    mode: Mode
    domain: Domain
    user_model: str = "qwen-max"
    reviewer_variant: str = "plain_checks_v1"
    seed: int = 42
    verification_budget: int = 1
    reviewer_temperature: float = 0.0
    reviewer_seed_mode: str = "fixed"
    reviewer_soft_label_output: bool = False
    reviewer_probability_aggregation: str = "vote_majority"
    reviewer_probability_threshold: float = 0.5
    reviewer_probability_early_stop: bool = False
    reviewer_excluded_from_pair_common: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 8192
    judge_max_completion_tokens: int = MAX_REVIEW_COMPLETION_TOKENS
    max_steps: int = 30
    request_timeout_seconds: float = 300.0
    pairing_mode: PairingMode = "independent"
    pair_run_id: str | None = None
    paired_task_ids: tuple[int, ...] | None = None
    include_tool_schemas_in_evidence: bool = True
    reviewer_candidate_effect_visibility: str = "full"
    reviewer_policy_presentation: str = "full"
    reviewer_transport_max_attempts: int = 1
    reviewer_transport_min_interval_seconds: float = 0.0
    reviewer_transport_retry_backoff_seconds: float = 30.0
    require_paired_cache_hits: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("baseline", "shadow"):
            raise ValueError("mode must be baseline or shadow")
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError("unsupported τ-bench domain")
        if self.mode == "baseline" and self.verification_budget != 1:
            raise ValueError("baseline requires verification_budget=1")
        if self.verification_budget not in (1, 3, 5, 7):
            raise ValueError("verification_budget must be 1, 3, 5, or 7")
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
        if type(self.include_tool_schemas_in_evidence) is not bool:
            raise TypeError("include_tool_schemas_in_evidence must be bool")
        if self.reviewer_candidate_effect_visibility not in {"full", "hidden"}:
            raise ValueError(
                "reviewer_candidate_effect_visibility must be full or hidden"
            )
        if self.reviewer_policy_presentation not in {"full", "action_focus_v1"}:
            raise ValueError(
                "reviewer_policy_presentation must be full or action_focus_v1"
            )
        if (
            type(self.reviewer_transport_max_attempts) is not int
            or not 1 <= self.reviewer_transport_max_attempts <= 10
        ):
            raise ValueError(
                "reviewer_transport_max_attempts must be an integer in [1, 10]"
            )
        if self.reviewer_transport_min_interval_seconds < 0:
            raise ValueError(
                "reviewer_transport_min_interval_seconds must be non-negative"
            )
        if self.reviewer_transport_retry_backoff_seconds < 0:
            raise ValueError(
                "reviewer_transport_retry_backoff_seconds must be non-negative"
            )
        if type(self.reviewer_excluded_from_pair_common) is not bool:
            raise TypeError("reviewer_excluded_from_pair_common must be bool")
        if type(self.require_paired_cache_hits) is not bool:
            raise TypeError("require_paired_cache_hits must be bool")
        if self.pairing_mode == "independent" and self.require_paired_cache_hits:
            raise ValueError("require_paired_cache_hits requires a paired run")
        if self.pairing_mode == "independent":
            if self.pair_run_id is not None or self.paired_task_ids is not None:
                raise ValueError("independent runs cannot carry pair identity")
        else:
            if not self.pair_run_id or not re.fullmatch(
                "[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.pair_run_id
            ):
                raise ValueError("paired runs require a safe pair_run_id")
            if not self.paired_task_ids:
                raise ValueError("paired runs require task IDs")
            if self.pairing_mode == "record" and self.mode != "baseline":
                raise ValueError("record mode is baseline-only")
            if self.pairing_mode == "replay" and self.mode != "shadow":
                raise ValueError("replay mode is shadow-only")

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
        if self.reviewer_policy_presentation == "full":
            value.pop("reviewer_policy_presentation")
        if self.reviewer_candidate_effect_visibility == "full":
            value.pop("reviewer_candidate_effect_visibility")
        if self.reviewer_transport_max_attempts == 1:
            value.pop("reviewer_transport_max_attempts")
        if self.reviewer_transport_min_interval_seconds == 0.0:
            value.pop("reviewer_transport_min_interval_seconds")
        if self.reviewer_transport_retry_backoff_seconds == 30.0:
            value.pop("reviewer_transport_retry_backoff_seconds")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {"schema": RUNNER_SCHEMA, "config": self.artifact_config}
        )

    @property
    def pair_common_fingerprint(self) -> str | None:
        if self.pairing_mode == "independent":
            return None
        value = self.artifact_config
        for key in (
            "mode",
            "pairing_mode",
            "reviewer_variant",
            "verification_budget",
            "reviewer_temperature",
            "reviewer_seed_mode",
            "reviewer_soft_label_output",
            "reviewer_probability_aggregation",
            "reviewer_probability_threshold",
            "reviewer_probability_early_stop",
            "include_tool_schemas_in_evidence",
            "reviewer_candidate_effect_visibility",
            "reviewer_policy_presentation",
            "reviewer_transport_max_attempts",
            "reviewer_transport_min_interval_seconds",
            "reviewer_transport_retry_backoff_seconds",
        ):
            value.pop(key, None)
        if self.reviewer_excluded_from_pair_common:
            value.pop("judge_model")
        return canonical_sha256({"schema": RUNNER_SCHEMA, "paired": value})

    @property
    def reviewer_prompt_sha256(self) -> str:
        return hashlib.sha256(
            reviewer_prompt(self.reviewer_variant).encode("utf-8")
        ).hexdigest()


class _ReplayPrefixBackend:
    """Replay the frozen baseline until rejection, then use uncached live calls."""

    def __init__(self, live: Any, cache_dir: Path) -> None:
        self.model = live.model
        self.config = live.config
        self._live = live
        self._replay = ReplayCacheBackend(live, cache_dir, replay_only=True)
        self._released = False
        self.last_cache_hit = False
        self.last_request_digest: str | None = None
        self.last_completion_digest: str | None = None

    def release_replay_only(self) -> None:
        self._released = True

    def complete(self, messages, tools=None, seed=None, json_mode=False):
        backend = self._live if self._released else self._replay
        completion = backend.complete(
            messages, tools=tools, seed=seed, json_mode=json_mode
        )
        self.last_cache_hit = not self._released
        self.last_request_digest = getattr(backend, "last_request_digest", None)
        self.last_completion_digest = getattr(backend, "last_completion_digest", None)
        return completion


class _MeteredBackend:
    def __init__(self, backend: Any, metrics: ParticipantMetrics) -> None:
        self.model = backend.model
        self.config = backend.config
        self._backend = backend
        self._metrics = metrics

    def complete(self, messages, tools=None, seed=None, json_mode=False):
        completion = self._backend.complete(
            messages, tools=tools, seed=seed, json_mode=json_mode
        )
        self._metrics.observe(completion, self._backend)
        return completion


def load_tasks(domain: Domain, task_ids: Sequence[int] | None = None) -> list[Any]:
    if domain == "retail":
        from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
    elif domain == "airline":
        from tau_bench.envs.airline.tasks_test import TASKS as tasks
    else:
        raise ValueError(f"unsupported domain: {domain}")
    requested = tuple(range(len(tasks))) if task_ids is None else tuple(task_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("task IDs must be unique")
    if any(
        (
            type(index) is not int or index < 0 or index >= len(tasks)
            for index in requested
        )
    ):
        raise ValueError("task ID is outside the official test split")
    return [tasks[index] for index in requested]


def _task_fingerprint(task: Any) -> str:
    return canonical_sha256(task.model_dump(mode="json"))


@functools.lru_cache(maxsize=None)
def data_provenance(domain: Domain) -> dict[str, Any]:
    require_clean_git_tree(TAU_BENCH_ROOT, subject="pinned upstream tau-bench tree")
    spec = importlib.util.find_spec("tau_bench")
    if spec is None or spec.origin is None:
        raise RuntimeError("tau_bench package is not importable")
    imported = Path(spec.origin).resolve()
    try:
        imported.relative_to(TAU_BENCH_ROOT.resolve())
    except ValueError:
        raise RuntimeError("tau_bench was imported from outside the pinned checkout")
    domain_root = TAU_BENCH_ROOT / "tau_bench" / "envs" / domain
    behavior_files = [
        *sorted(domain_root.rglob("*.py")),
        *sorted((domain_root / "data").glob("*.json")),
        domain_root / "wiki.md",
        TAU_BENCH_ROOT / "tau_bench" / "envs" / "base.py",
        TAU_BENCH_ROOT / "tau_bench" / "envs" / "user.py",
        TAU_BENCH_ROOT / "tau_bench" / "types.py",
        TAU_BENCH_ROOT / "setup.py",
    ]
    return {
        "source": "sierra-research/tau-bench",
        "base_upstream_git_head": "59a200c6d575d595120f1cb70fea53cef0632f6b",
        "bugfixed_git_head": git_head(TAU_BENCH_ROOT),
        "domain": domain,
        "task_split": "test",
        "official_task_count": TASK_COUNTS[domain],
        "imported_from": str(imported.relative_to(REPO_ROOT)),
        "files": {
            str(path.relative_to(TAU_BENCH_ROOT)): sha256_file(path)
            for path in behavior_files
        },
        "semantic_repairs": [
            {
                "name": "persist_requested_cabin_on_flight_update",
                "upstream_pr": 23,
                "scope": "one assignment plus regression test",
            }
        ],
    }


@functools.lru_cache(maxsize=1)
def implementation_provenance() -> dict[str, Any]:
    roots = (
        REPO_ROOT / "shadow-verifier" / "src",
        REPO_ROOT / "workflow" / "tau_bench",
    )
    files: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.relative_to(root).parts:
                continue
            files[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    runtime = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    return {
        "files": files,
        "runtime": runtime,
        "fingerprint": canonical_sha256({"files": files, "runtime": runtime}),
    }


def ensure_pair_manifest(
    output_root: Path, configs: Sequence[TauBenchConfig]
) -> dict[str, Any]:
    if not configs or any((config.pairing_mode == "independent" for config in configs)):
        raise ValueError("pair manifest requires paired configs")
    first = configs[0]
    if any(
        (
            config.pair_common_fingerprint != first.pair_common_fingerprint
            for config in configs
        )
    ):
        raise ValueError("paired configs do not share a common fingerprint")
    payload = {
        "schema": "shadow-tau-bench-pair-v1",
        "benchmark": "tau-bench",
        "domain": first.domain,
        "task_split": "test",
        "pair_run_id": first.pair_run_id,
        "task_ids": list(first.paired_task_ids or ()),
        "pair_common_fingerprint": first.pair_common_fingerprint,
        **(
            {"shadow_reviewer_scoring": [None for config in configs if False]}
            if any((False for config in configs))
            else {}
        ),
        "data_provenance_fingerprint": canonical_sha256(data_provenance(first.domain)),
        "implementation_fingerprint": implementation_provenance()["fingerprint"],
    }
    payload["manifest_fingerprint"] = canonical_sha256(payload)
    path = (
        output_root
        / "_pair_manifests"
        / "tau_bench"
        / first.domain
        / str(first.pair_run_id)
        / "manifest.json"
    )
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != payload:
            raise RuntimeError("pair_run_id already belongs to another experiment")
    else:
        atomic_json_dump(path, payload)
    return payload


def _cache_namespace(config: TauBenchConfig) -> str:
    return canonical_sha256(
        {
            "pair_common_fingerprint": config.pair_common_fingerprint,
            "data": canonical_sha256(data_provenance(config.domain)),
            "implementation": implementation_provenance()["fingerprint"],
        }
    )


def _task_cache_dir(
    output_root: Path, config: TauBenchConfig, task_id: int, role: str
) -> Path:
    return (
        output_root
        / "_paired_completion_cache"
        / "tau_bench"
        / config.domain
        / str(config.pair_run_id)
        / _cache_namespace(config)
        / f"task-{task_id:03d}"
        / role
    )


def _live_backend(model: str, config: TauBenchConfig, *, judge: bool = False):
    return hosted_backend(
        model=model,
        default_factory=DashScopeBackend,
        role="verifier"
        if judge
        else "user_simulator"
        if model == config.user_model
        else "producer",
        config=DashScopeConfig(
            temperature=config.reviewer_temperature if judge else config.temperature,
            top_p=1.0 if judge else config.top_p,
            max_completion_tokens=config.judge_max_completion_tokens
            if judge
            else config.max_completion_tokens,
            enable_thinking=False,
            timeout_seconds=config.request_timeout_seconds,
            transport_max_attempts=config.reviewer_transport_max_attempts
            if judge
            else 1,
            transport_min_interval_seconds=config.reviewer_transport_min_interval_seconds
            if judge
            else 0.0,
            transport_retry_backoff_seconds=config.reviewer_transport_retry_backoff_seconds
            if judge
            else 30.0,
        ),
    )


def _participant_backend(
    model: str,
    config: TauBenchConfig,
    *,
    output_root: Path,
    task_id: int,
    role: Literal["producer", "user"],
    continuation_cache_root: Path | None = None,
):
    live = _live_backend(model, config)
    if continuation_cache_root is not None:
        from workflow.experiment_replay import OccurrenceCacheBackend

        live = OccurrenceCacheBackend(live, continuation_cache_root / role)
    if config.pairing_mode == "independent":
        return live
    cache_dir = _task_cache_dir(output_root, config, task_id, role)
    if config.pairing_mode == "record":
        return ReplayCacheBackend(
            live, cache_dir, replay_only=config.require_paired_cache_hits
        )
    return _ReplayPrefixBackend(live, cache_dir)


def _make_environment(domain: Domain, task_id: int):
    if domain == "retail":
        from tau_bench.envs.retail import MockRetailDomainEnv

        environment = MockRetailDomainEnv(user_strategy="human", task_index=task_id)
    else:
        from tau_bench.envs.airline import MockAirlineDomainEnv

        environment = MockAirlineDomainEnv(user_strategy="human", task_index=task_id)
    known = frozenset(environment.tools_map)
    classified = MUTATING_TOOLS[domain] | READ_ONLY_TOOLS[domain]
    if known != classified or MUTATING_TOOLS[domain] & READ_ONLY_TOOLS[domain]:
        raise RuntimeError("τ-bench tool mutation classification is incomplete")
    return environment


def _execute_tool(environment: Any, action: Any) -> tuple[str, bool]:
    environment.actions.append(action)
    try:
        observation = environment.tools_map[action.name].invoke(
            data=environment.data, **action.kwargs
        )
    except Exception as exc:
        observation = f"Error: {exc}"
    return (str(observation), action.name in environment.terminate_tools)


def _assistant_message(
    completion: Any,
) -> tuple[dict[str, Any], TauBenchToolAction | None]:
    if completion.tool_calls:
        call = completion.tool_calls[0]
        arguments = json.loads(
            json.dumps(
                call.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        if type(arguments) is not dict:
            raise TypeError("tool-call arguments must be a JSON object")
        action = TauBenchToolAction(call.id, call.name, arguments)
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                arguments,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            },
            action,
        )
    return ({"role": "assistant", "content": completion.content}, None)


def result_path(output_root: Path, config: TauBenchConfig, task_id: int) -> Path:
    path = (
        output_root
        / "tau_bench"
        / config.domain
        / config.producer_model.replace("/", "_")
        / config.mode
    )
    if config.mode == "shadow":
        path /= f"variant-{config.reviewer_variant}"
        path /= f"verification-budget-{config.verification_budget}"
        if config.reviewer_policy_presentation != "full":
            path /= f"policy-{config.reviewer_policy_presentation.replace('_', '-')}"
        if config.reviewer_candidate_effect_visibility != "full":
            path /= "candidate-effect-hidden"
    path /= f"judge-{config.judge_model.replace('/', '_')}"
    path /= f"s{config.seed}"
    if config.pair_run_id:
        path /= f"pair-{config.pair_run_id}"
    return path / f"task-{task_id:03d}" / "result.json"


def run_task(
    *,
    task_id: int,
    config: TauBenchConfig,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    resume: bool = True,
    first_rejection_diagnostic: Any | None = None,
    continuation_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Run one official task. This workflow intentionally has no unit-test hook."""
    official_task = load_tasks(config.domain, (task_id,))[0]
    task_fingerprint = _task_fingerprint(official_task)
    provenance = data_provenance(config.domain)
    implementation = implementation_provenance()
    identity = {
        "schema": RUNNER_SCHEMA,
        "benchmark": "tau-bench",
        "domain": config.domain,
        "task_split": "test",
        "task_id": task_id,
        "official_task_fingerprint": task_fingerprint,
        "config_fingerprint": config.fingerprint,
        "data_provenance_fingerprint": canonical_sha256(provenance),
        "implementation_fingerprint": implementation["fingerprint"],
        "reviewer_prompt_sha256": config.reviewer_prompt_sha256,
    }
    path = result_path(output_root, config, task_id)
    if first_rejection_diagnostic is not None:
        if (
            config.mode != "shadow"
            or config.pairing_mode != "replay"
            or config.verification_budget != 1
        ):
            raise ValueError("first-rejection diagnostic requires paired Shadow B1")
        identity["first_rejection_diagnostic"] = first_rejection_diagnostic.metadata
    if resume and path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if all((prior.get(key) == value for key, value in identity.items())):
            if prior.get("status") == "complete":
                return prior
        elif prior.get("status") not in {"running", "failed"}:
            raise RuntimeError("result path belongs to another experiment")
    atomic_json_dump(path, {"status": "running", **identity})
    environment = _make_environment(config.domain, task_id)
    producer_backend = _participant_backend(
        config.producer_model,
        config,
        output_root=output_root,
        task_id=task_id,
        role="producer",
        continuation_cache_root=continuation_cache_root,
    )
    user_backend = _participant_backend(
        config.user_model,
        config,
        output_root=output_root,
        task_id=task_id,
        role="user",
        continuation_cache_root=continuation_cache_root,
    )
    user = BackendUserSimulator(
        instruction=official_task.instruction, backend=user_backend, seed=config.seed
    )
    environment.user = user
    reset = environment.reset(task_index=task_id)
    producer_messages: list[dict[str, Any]] = [
        {"role": "system", "content": environment.wiki},
        {"role": "user", "content": reset.observation},
    ]
    canonical_messages = copy.deepcopy(producer_messages)
    producer_metrics = ParticipantMetrics()
    judge_metrics = ParticipantMetrics()
    gate_events: list[dict[str, Any]] = []
    first_rejection = False
    termination_reason = "max_steps"
    shadow = None
    if config.mode == "shadow":
        executor = TauBenchForkExecutor(
            environment,
            policy=environment.wiki,
            context_provider=lambda: copy.deepcopy(canonical_messages[1:]),
            tool_schemas=environment.tools_info,
            include_tool_schemas_in_evidence=config.include_tool_schemas_in_evidence,
            policy_presentation=config.reviewer_policy_presentation,
        )
        judge_live = _live_backend(config.judge_model, config, judge=True)
        if continuation_cache_root is not None:
            from workflow.experiment_replay import OccurrenceCacheBackend

            judge_live = OccurrenceCacheBackend(
                judge_live, continuation_cache_root / "reviewer"
            )
        judge_backend = _MeteredBackend(judge_live, judge_metrics)
        reviewer_runtime_model = config.judge_model
        if reviewer_runtime_model is None:
            raise RuntimeError("shadow reviewer model identity is missing")
        reviewer_factory = SemanticReviewerFactory(
            lambda _model: judge_backend,
            model=reviewer_runtime_model,
            config=SemanticReviewerConfig(
                seed=config.seed,
                system_prompt=reviewer_prompt(config.reviewer_variant),
                soft_label_output=config.reviewer_soft_label_output,
                candidate_effect_visibility=config.reviewer_candidate_effect_visibility,
            ),
            sample_seed_mode=config.reviewer_seed_mode,
        )
        shadow = ShadowVerifier(
            executor,
            first_rejection_diagnostic.factory
            if first_rejection_diagnostic is not None
            else reviewer_factory,
            verification_budget=config.verification_budget,
            probability_aggregation=config.reviewer_probability_aggregation,
            probability_threshold=config.reviewer_probability_threshold,
            probability_early_stop=config.reviewer_probability_early_stop,
        )
    started = time.perf_counter()
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    for step_index in range(config.max_steps):
        completion = producer_backend.complete(
            messages=producer_messages, tools=environment.tools_info, seed=config.seed
        )
        producer_metrics.observe(completion, producer_backend)
        assistant_message, proposed = _assistant_message(completion)
        if proposed is None:
            content = assistant_message.get("content")
            if not isinstance(content, str):
                raise RuntimeError("producer returned neither text nor a tool call")
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
            environment.actions.append(action)
            observation = user.step(content)
            user_message = {"role": "user", "content": observation}
            producer_messages.extend([assistant_message, user_message])
            canonical_messages.extend(copy.deepcopy([assistant_message, user_message]))
            if "###STOP###" in observation:
                termination_reason = "user_stop"
                break
            continue
        action = Action(name=proposed.name, kwargs=copy.deepcopy(proposed.arguments))
        is_mutating = proposed.name in MUTATING_TOOLS[config.domain]
        published = True
        if (
            shadow is not None
            and is_mutating
            and (
                not (
                    first_rejection_diagnostic is not None
                    and first_rejection_diagnostic.finished
                )
            )
        ):
            result = shadow.step(f"tau-bench:{step_index}:{proposed.call_id}", proposed)
            observation = result.observation
            published = isinstance(result, Published)
            record = shadow.audit_records[-1]
            if first_rejection_diagnostic is not None:
                first_rejection_diagnostic.check_record(record)
            gate_events.append(
                {
                    "action": proposed.as_json(),
                    "outcome": record.outcome,
                    "decision": record.decision.as_json(),
                    "verification_budget": len(record.review_decisions),
                    "accept_votes": sum(
                        (item.accept for item in record.review_decisions)
                    ),
                    "reject_votes": sum(
                        (not item.accept for item in record.review_decisions)
                    ),
                    "review_decisions": [
                        {"sample_index": index, **item.as_json()}
                        for index, item in enumerate(record.review_decisions)
                    ],
                    "evidence_sha256": record.evidence_sha256,
                    "stage_seconds": record.stage_seconds,
                    "review_seconds": record.review_seconds,
                    "settle_seconds": record.settle_seconds,
                    "total_seconds": record.total_seconds,
                }
            )
            if published:
                environment.actions.append(action)
            elif not first_rejection:
                first_rejection = True
                producer_backend.release_replay_only()
                user_backend.release_replay_only()
        else:
            observation, _ = _execute_tool(environment, action)
        tool_message = {
            "role": "tool",
            "tool_call_id": proposed.call_id,
            "name": proposed.name,
            "content": observation,
        }
        producer_messages.extend([assistant_message, tool_message])
        if published:
            canonical_messages.extend(copy.deepcopy([assistant_message, tool_message]))
        if published and proposed.name in environment.terminate_tools:
            termination_reason = "transfer_to_human"
            break
    elapsed = time.perf_counter() - started
    if first_rejection_diagnostic is not None:
        first_rejection_diagnostic.assert_complete()
    evaluation = evaluate(environment, official_task)
    metrics = {
        "elapsed_s": round(elapsed, 6),
        "producer": producer_metrics.as_json(),
        "user": user.metrics.as_json(),
        "judge": judge_metrics.as_json(),
        "producer_message_count": len(producer_messages),
        "canonical_message_count": len(canonical_messages),
        "shadow_reviewed_actions": len(gate_events),
        "shadow_rejected_actions": sum(
            (event["outcome"] == "rejected" for event in gate_events)
        ),
        "shadow_published_actions": sum(
            (event["outcome"] == "published" for event in gate_events)
        ),
        "candidate_policy_test_reports": 0,
        **{},
    }
    payload = {
        "status": "complete",
        **identity,
        "mode": config.mode,
        "producer_model": config.producer_model,
        "judge_model": config.judge_model if config.mode == "shadow" else None,
        "user_model": config.user_model,
        "reviewer_variant": config.reviewer_variant,
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
        "seed": config.seed,
        "pair_run_id": config.pair_run_id,
        "pairing_mode": config.pairing_mode,
        "pair_common_fingerprint": config.pair_common_fingerprint,
        "config": config.artifact_config,
        "data_provenance": provenance,
        "implementation_provenance": implementation,
        "code_provenance": {
            "implementation_git_head": git_head(REPO_ROOT),
            "tau_bench_git_head": git_head(TAU_BENCH_ROOT),
        },
        "integrity": {
            "official_test_split": True,
            "official_task_unchanged": True,
            "official_environment_with_documented_pr23_repair": True,
            "unit_tests_in_verifier_context": False,
            "gold_available_to_producer": False,
            "gold_available_to_shadow_judge": False,
            "pure_reads_gated": False,
            "exact_baseline_replay_until_first_rejection": config.pairing_mode
            == "replay",
        },
        "termination_reason": termination_reason,
        "official_reward": evaluation["reward"],
        "evaluation": evaluation,
        "metrics": metrics,
        "gate_events": gate_events,
        "canonical_actions": [
            action.model_dump(mode="json") for action in environment.actions
        ],
        "canonical_messages": canonical_messages,
        "producer_messages": producer_messages,
    }
    atomic_json_dump(path, payload)
    return payload


def summarize_cell(
    output_root: Path, config: TauBenchConfig, task_ids: Sequence[int]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for task_id in task_ids:
        path = result_path(output_root, config, task_id)
        if not path.exists():
            missing.append(task_id)
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("status") != "complete"
            or row.get("config_fingerprint") != config.fingerprint
        ):
            missing.append(task_id)
            continue
        rows.append(row)
    clean = [row for row in rows if row["evaluation"]["gold_replay_clean"]]
    return {
        "schema": "shadow-tau-bench-cell-summary-v1",
        "status": "complete" if not missing else "incomplete",
        "domain": config.domain,
        "mode": config.mode,
        "verification_budget": config.verification_budget,
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
        "pair_run_id": config.pair_run_id,
        "requested_tasks": len(task_ids),
        "completed_tasks": len(rows),
        "missing_task_ids": missing,
        "successes": sum((row["official_reward"] == 1.0 for row in rows)),
        "success_rate": sum((row["official_reward"] == 1.0 for row in rows)) / len(rows)
        if rows
        else None,
        "gold_replay_clean_tasks": len(clean),
        "clean_successes": sum((row["official_reward"] == 1.0 for row in clean)),
        "clean_success_rate": sum((row["official_reward"] == 1.0 for row in clean))
        / len(clean)
        if clean
        else None,
        "shadow_rejected_actions": sum(
            (row["metrics"]["shadow_rejected_actions"] for row in rows)
        ),
        "judge_api_calls": sum((row["metrics"]["judge"]["api_calls"] for row in rows)),
        **{},
        "producer_api_calls": sum(
            (row["metrics"]["producer"]["api_calls"] for row in rows)
        ),
        "producer_replay_cache_hits": sum(
            (row["metrics"]["producer"]["replay_cache_hits"] for row in rows)
        ),
        "user_api_calls": sum((row["metrics"]["user"]["api_calls"] for row in rows)),
        "user_replay_cache_hits": sum(
            (row["metrics"]["user"]["replay_cache_hits"] for row in rows)
        ),
    }
