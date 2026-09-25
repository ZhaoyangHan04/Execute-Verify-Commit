"""Reproducible non-leaderboard scoring for paired STATE-Bench artifacts.

The pinned benchmark's state requirements are evaluated deterministically.
Non-state requirements use the pinned domain prompt with a separately declared
DashScope judge.  Original trajectories are read-only; all scores and judge
cache entries live outside the pair directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from shadow_verifier.backends.dashscope import DashScopeBackend, DashScopeConfig
from shadow_verifier.backends.replay import ReplayCacheBackend
from shadow_verifier.experiments.artifacts import (
    atomic_json_dump,
    canonical_sha256,
    sha256_file,
)
from state_bench.domain import get_domain_config
from state_bench.paths import domain_tasks_dir
from state_bench.protocol import load_default_protocol, load_split_task_ids
from state_bench.schemas import StateDiff, TaskDefinition
from state_bench.scoring import (
    TaskRequirementsJudge,
    build_task_requirements_prompt,
    evaluate_state_requirements,
    evaluate_task_requirements_empty,
)


SCHEMA = "state-bench-shadow-local-score-v1"
PROTOCOL_LABEL = "custom_qwen_task_requirements_nonleaderboard_v1"
DOMAINS = ("customer_support", "shopping_assistant", "travel")
TERMINAL_STATUSES = {"complete", "failed"}


@dataclass(frozen=True)
class ScoreConfig:
    evaluator_model: str = "qwen3.8-max"
    seed: int = 42
    workers: int = 4
    max_completion_tokens: int = 8192
    timeout_seconds: float = 300.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.evaluator_model.strip():
            raise ValueError("evaluator_model must be non-empty")
        if type(self.seed) is not int:
            raise TypeError("seed must be int")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


@dataclass(frozen=True)
class _TrajectoryInput:
    producer_model: str
    domain: str
    task_id: str
    arm: str
    pair_id: str
    pair_status: str
    path: Path | None
    pair_dir: Path
    reviewer_variant: str | None
    reviewed_batches: int
    rejected_batches: int
    runtime_error_type: str | None = None


class _BackendJsonClient:
    """Adapt the strict completion backend to the pinned judge interface."""

    def __init__(self, backend: ReplayCacheBackend, *, seed: int) -> None:
        self.backend = backend
        self.seed = seed
        self.last_completion: Any | None = None

    def complete_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        del max_tokens
        if reasoning_effort is not None:
            raise ValueError("the frozen local protocol does not use reasoning_effort")
        completion = self.backend.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            seed=self.seed,
            json_mode=True,
        )
        self.last_completion = completion
        if not isinstance(completion.content, str):
            raise ValueError("JSON judge returned no text content")
        value = json.loads(completion.content)
        if not isinstance(value, dict):
            raise ValueError("JSON judge response must be an object")
        return value


_thread_local = threading.local()


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _domain_from_task_root(name: str) -> str:
    matches = [domain for domain in DOMAINS if name.startswith(f"{domain}-")]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve domain from pair task root: {name!r}")
    return matches[0]


def _is_retryable_infrastructure_failure(status: Mapping[str, Any]) -> bool:
    review_failure = status.get("review_failure")
    return (
        status.get("status") == "failed"
        and (
            status.get("failure_classification") == "review_infrastructure"
            or (
                status.get("error_type")
                == "StateBenchReviewInfrastructureError"
                and isinstance(review_failure, Mapping)
                and review_failure.get("code") == "review_unavailable"
            )
        )
    )


def _discover_model_inputs(
    model_dir: Path,
    *,
    task_split: str,
    count_terminal_missing_as_zero: bool = False,
) -> list[_TrajectoryInput]:
    protocol = load_default_protocol()
    expected = {
        (domain, task_id)
        for domain in DOMAINS
        for task_id in load_split_task_ids(domain, task_split, protocol.split_version)
    }
    candidates: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for status_path in sorted(model_dir.glob("*/*/status.json")):
        status = _strict_json(status_path)
        pair_status = status.get("status")
        if pair_status not in TERMINAL_STATUSES:
            raise RuntimeError(f"non-terminal pair status at {status_path}: {pair_status!r}")
        pair_dir = status_path.parent
        baseline_path = pair_dir / "baseline" / "trajectory.json"
        if not baseline_path.is_file():
            if count_terminal_missing_as_zero and pair_status == "failed":
                # A baseline-arm failure can happen before trajectory persistence.
                # The matrix remains the authoritative task identity in that case.
                continue
            raise RuntimeError(f"terminal pair has no baseline trajectory: {pair_dir}")
        task_id = _strict_json(baseline_path).get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"baseline trajectory has invalid task_id: {baseline_path}")
        identity = (_domain_from_task_root(pair_dir.parent.name), task_id)
        if identity not in expected:
            raise ValueError(f"pair is outside {task_split!r} split: {identity}")
        candidates[identity].append((status_path, status))

    selected_status_paths: list[Path] = []
    for identity, attempts in candidates.items():
        if len(attempts) == 1:
            selected_status_paths.append(attempts[0][0])
            continue
        complete = [item for item in attempts if item[1].get("status") == "complete"]
        superseded = [item for item in attempts if item[1].get("status") != "complete"]
        if len(complete) != 1 or not superseded or not all(
            _is_retryable_infrastructure_failure(status) for _path, status in superseded
        ):
            raise ValueError(f"ambiguous multiple terminal attempts for one task: {identity}")
        selected_status_paths.append(complete[0][0])

    found: dict[tuple[str, str], _TrajectoryInput] = {}
    result: list[_TrajectoryInput] = []
    for status_path in sorted(selected_status_paths):
        pair_dir = status_path.parent
        status = _strict_json(status_path)
        pair_status = status.get("status")
        if pair_status not in TERMINAL_STATUSES:
            raise RuntimeError(f"non-terminal pair status at {status_path}: {pair_status!r}")
        summary_path = pair_dir / "pair_summary.json"
        baseline_path = pair_dir / "baseline" / "trajectory.json"
        if not baseline_path.is_file():
            raise RuntimeError(f"terminal pair has no baseline trajectory: {pair_dir}")
        baseline = _strict_json(baseline_path)
        task_id = baseline.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"baseline trajectory has invalid task_id: {baseline_path}")
        domain = _domain_from_task_root(pair_dir.parent.name)
        identity = (domain, task_id)
        if identity not in expected:
            raise ValueError(f"pair is outside {task_split!r} split: {identity}")
        if identity in found:
            raise ValueError(f"multiple terminal attempts for one task: {identity}")

        pair_id = status.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"status has invalid pair_id: {status_path}")
        reviewer_variant: str | None = None
        reviewed_batches = 0
        rejected_batches = 0
        shadow_path: Path | None = None
        runtime_error_type: str | None = None
        if pair_status == "complete":
            if not summary_path.is_file():
                raise RuntimeError(f"complete pair has no summary: {pair_dir}")
            summary = _strict_json(summary_path)
            summary_identity = summary.get("pair_identity", {})
            if summary_identity.get("domain") != domain or summary_identity.get("task_id") != task_id:
                raise ValueError(f"pair summary identity mismatch: {summary_path}")
            config = summary.get("config", {})
            reviewer_variant = config.get("reviewer_variant")
            metrics = summary.get("shadow", {}).get("execution_metrics", {})
            reviewed_batches = int(metrics.get("reviewed_batches", 0))
            rejected_batches = int(metrics.get("rejected_batches", 0))
            shadow_path = pair_dir / "shadow" / "trajectory.json"
            if not shadow_path.is_file():
                raise RuntimeError(f"complete pair has no shadow trajectory: {pair_dir}")
        else:
            runtime_error_type = status.get("error_type")
            audit_path = pair_dir / "shadow" / "review_audit.json"
            if audit_path.is_file():
                audit = _strict_json(audit_path)
                reviewed_batches = int(audit.get("record_count", 0))
                records = audit.get("records", [])
                if isinstance(records, list):
                    rejected_batches = sum(
                        isinstance(row, Mapping)
                        and isinstance(row.get("decision"), Mapping)
                        and row["decision"].get("accept") is False
                        for row in records
                    )

        base_input = _TrajectoryInput(
            producer_model=model_dir.name,
            domain=domain,
            task_id=task_id,
            arm="baseline",
            pair_id=pair_id,
            pair_status=pair_status,
            path=baseline_path,
            pair_dir=pair_dir,
            reviewer_variant=reviewer_variant,
            reviewed_batches=reviewed_batches,
            rejected_batches=rejected_batches,
            runtime_error_type=runtime_error_type,
        )
        found[identity] = base_input
        result.append(base_input)
        result.append(
            _TrajectoryInput(
                **{
                    **asdict(base_input),
                    "arm": "shadow",
                    "path": shadow_path,
                }
            )
        )

    missing = sorted(expected - set(found))
    if missing and count_terminal_missing_as_zero:
        matrix_path = model_dir / "matrix_status.json"
        matrix = _strict_json(matrix_path)
        matrix_config = matrix.get("config")
        jobs = matrix.get("jobs")
        if not isinstance(matrix_config, Mapping) or not isinstance(jobs, Mapping):
            raise ValueError(f"malformed terminal matrix: {matrix_path}")
        if matrix_config.get("split") != task_split:
            raise ValueError(f"matrix split mismatch: {matrix_path}")
        reviewer_variant = matrix_config.get("reviewer_variant")
        unresolved: list[tuple[str, str]] = []
        for domain, task_id in missing:
            job = jobs.get(f"{domain}:{task_id}")
            if not isinstance(job, Mapping) or job.get("status") != "failed":
                unresolved.append((domain, task_id))
                continue
            synthetic_pair_id = f"matrix-terminal-failure:{domain}:{task_id}"
            base_input = _TrajectoryInput(
                producer_model=model_dir.name,
                domain=domain,
                task_id=task_id,
                arm="baseline",
                pair_id=synthetic_pair_id,
                pair_status="failed",
                path=None,
                pair_dir=model_dir,
                reviewer_variant=(
                    reviewer_variant if isinstance(reviewer_variant, str) else None
                ),
                reviewed_batches=0,
                rejected_batches=0,
                runtime_error_type="TerminalFailureBeforeBaselinePersistence",
            )
            found[(domain, task_id)] = base_input
            result.append(base_input)
            result.append(
                _TrajectoryInput(
                    **{
                        **asdict(base_input),
                        "arm": "shadow",
                    }
                )
            )
        missing = unresolved
    extra = sorted(set(found) - expected)
    if missing or extra:
        raise RuntimeError(
            f"incomplete model inventory for {model_dir.name}: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return result


def _load_task(domain: str, task_id: str) -> TaskDefinition:
    return TaskDefinition.load(domain_tasks_dir(domain) / f"{task_id}.json")


def _state_diff(raw: object, *, path: Path) -> StateDiff:
    if not isinstance(raw, dict) or not {"created", "modified", "deleted"}.issubset(raw):
        raise ValueError(f"malformed state_diff: {path}")
    return StateDiff(
        created=raw.get("created", {}),
        modified=raw.get("modified", {}),
        deleted=raw.get("deleted", {}),
    )


def _tool_calls(conversation: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in conversation:
        value = message.get("tool_calls")
        if isinstance(value, list):
            calls.extend(value)
    return calls


def _judge_key(
    *,
    domain: str,
    prompt: str,
    system_prompt: str,
    config: ScoreConfig,
) -> str:
    return canonical_sha256(
        {
            "protocol": PROTOCOL_LABEL,
            "domain": domain,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": config.evaluator_model,
            "seed": config.seed,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion_tokens": config.max_completion_tokens,
            "enable_thinking": False,
        }
    )


def _thread_backend(config: ScoreConfig, cache_dir: Path) -> ReplayCacheBackend:
    key = (config, cache_dir.resolve())
    current = getattr(_thread_local, "backend", None)
    current_key = getattr(_thread_local, "backend_key", None)
    if current is None or current_key != key:
        raw = DashScopeBackend(
            model=config.evaluator_model,
            config=DashScopeConfig(
                temperature=0.0,
                top_p=1.0,
                max_completion_tokens=config.max_completion_tokens,
                enable_thinking=False,
                timeout_seconds=config.timeout_seconds,
            ),
        )
        current = ReplayCacheBackend(raw, cache_dir=cache_dir)
        _thread_local.backend = current
        _thread_local.backend_key = key
    return current


def _evaluate_nonempty_requirements(
    *,
    task: TaskDefinition,
    conversation: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    domain: str,
    config: ScoreConfig,
    cache_dir: Path,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for _attempt in range(1, config.max_attempts + 1):
        backend = _thread_backend(config, cache_dir)
        client = _BackendJsonClient(backend, seed=config.seed)
        judge = TaskRequirementsJudge(
            client=client,
            prompts_dir=get_domain_config(domain).prompts_dir,
            system_prompt=get_domain_config(domain).judge_system_prompt,
            reasoning_effort=None,
        )
        result = judge.evaluate(
            task=task,
            conversation=conversation,
            tool_calls=tool_calls,
            state_diff=StateDiff(),
        )
        if result is not None and client.last_completion is not None:
            completion = client.last_completion
            return {
                "score": result.to_dict(),
                "judge_audit": {
                    "cache_hit": backend.last_cache_hit,
                    "request_sha256": backend.last_request_digest,
                    "completion_sha256": backend.last_completion_digest,
                    "response_model": completion.model,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                },
            }
        last_error = RuntimeError("task-requirements judge returned no valid result")
    raise RuntimeError("task-requirements judge exhausted retries") from last_error


def _mcnemar_exact(gains: int, harms: int) -> float:
    discordant = gains + harms
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, harms) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_metrics(records: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    by_pair: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for record in records:
        score = record.get(score_field)
        if type(score) is int:
            key = (record["producer_model"], record["domain"], record["task_id"])
            by_pair[key][record["arm"]] = score
    complete = [arms for arms in by_pair.values() if set(arms) == {"baseline", "shadow"}]
    baseline_pass = sum(arms["baseline"] for arms in complete)
    shadow_pass = sum(arms["shadow"] for arms in complete)
    gains = sum(arms == {"baseline": 0, "shadow": 1} for arms in complete)
    harms = sum(arms == {"baseline": 1, "shadow": 0} for arms in complete)
    n = len(complete)
    return {
        "pairs": n,
        "baseline_pass": baseline_pass,
        "shadow_pass": shadow_pass,
        "baseline_rate": baseline_pass / n if n else None,
        "shadow_rate": shadow_pass / n if n else None,
        "delta": (shadow_pass - baseline_pass) / n if n else None,
        "delta_pp": 100.0 * (shadow_pass - baseline_pass) / n if n else None,
        "gains": gains,
        "harms": harms,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(gains, harms),
    }


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": records}
    for model in sorted({row["producer_model"] for row in records}):
        groups[f"model:{model}"] = [row for row in records if row["producer_model"] == model]
        for domain in DOMAINS:
            groups[f"model:{model}/domain:{domain}"] = [
                row
                for row in records
                if row["producer_model"] == model and row["domain"] == domain
            ]
    return {
        name: {
            "state_requirements": _paired_metrics(values, "state_pass"),
            "task_completion": _paired_metrics(values, "task_completion_pass"),
        }
        for name, values in groups.items()
    }


def score_experiment(
    *,
    experiment_root: Path,
    output_path: Path,
    cache_dir: Path,
    config: ScoreConfig,
    task_split: str = "test",
    producer_models: tuple[str, ...] | None = None,
    count_terminal_missing_as_zero: bool = False,
) -> dict[str, Any]:
    if task_split != "test":
        raise ValueError("local experiment scorer currently freezes task_split='test'")
    model_dirs = (
        [experiment_root / model for model in producer_models]
        if producer_models
        else sorted(path for path in experiment_root.iterdir() if path.is_dir() and path.name.startswith("qwen"))
    )
    if not model_dirs:
        raise ValueError("no producer model directories found")
    for path in model_dirs:
        if not path.is_dir():
            raise FileNotFoundError(path)

    inputs = [
        item
        for model_dir in model_dirs
        for item in _discover_model_inputs(
            model_dir,
            task_split=task_split,
            count_terminal_missing_as_zero=count_terminal_missing_as_zero,
        )
    ]
    records: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    pending_record_indices: dict[str, list[int]] = defaultdict(list)

    for item in inputs:
        base = {
            "producer_model": item.producer_model,
            "domain": item.domain,
            "task_id": item.task_id,
            "pair_id": item.pair_id,
            "pair_status": item.pair_status,
            "arm": item.arm,
            "reviewer_variant": item.reviewer_variant,
            "reviewed_batches": item.reviewed_batches,
            "rejected_batches": item.rejected_batches,
            "runtime_error_type": item.runtime_error_type,
        }
        if item.path is None:
            records.append(
                {
                    **base,
                    "trajectory_path": None,
                    "trajectory_sha256": None,
                    "state_requirements": None,
                    "state_pass": 0,
                    "task_requirements": {"status": "not_evaluated_runtime_failure"},
                    "task_requirements_pass": 0,
                    "task_completion_pass": 0,
                }
            )
            continue

        trajectory = _strict_json(item.path)
        task = _load_task(item.domain, item.task_id)
        state_result = evaluate_state_requirements(
            task,
            _state_diff(trajectory.get("state_diff"), path=item.path),
        )
        if state_result is None:
            raise RuntimeError(f"state scorer returned None: {item.path}")
        record: dict[str, Any] = {
            **base,
            "trajectory_path": str(item.path.relative_to(experiment_root)),
            "trajectory_sha256": sha256_file(item.path),
            "state_requirements": state_result.to_dict(),
            "state_pass": state_result.score,
            "task_requirements": None,
            "task_requirements_pass": None,
            "task_completion_pass": 0,
        }
        records.append(record)
        index = len(records) - 1
        if state_result.score == 0:
            record["task_requirements"] = {"status": "not_evaluated_state_failure"}
            record["task_requirements_pass"] = None
            continue
        empty = evaluate_task_requirements_empty(task)
        if empty is not None:
            record["task_requirements"] = {
                "status": "deterministic_empty_requirements",
                "score": empty.to_dict(),
            }
            record["task_requirements_pass"] = empty.score
            record["task_completion_pass"] = int(empty.score == 1)
            continue
        conversation = trajectory.get("conversation")
        if not isinstance(conversation, list) or not all(isinstance(row, dict) for row in conversation):
            raise ValueError(f"malformed conversation: {item.path}")
        domain_config = get_domain_config(item.domain)
        prompt = build_task_requirements_prompt(
            task=task,
            conversation=conversation,
            tool_calls=_tool_calls(conversation),
            prompts_dir=domain_config.prompts_dir,
        )
        key = _judge_key(
            domain=item.domain,
            prompt=prompt,
            system_prompt=domain_config.judge_system_prompt,
            config=config,
        )
        pending_record_indices[key].append(index)
        pending.setdefault(
            key,
            {
                "task": task,
                "conversation": conversation,
                "tool_calls": _tool_calls(conversation),
                "domain": item.domain,
            },
        )

    completed: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                _evaluate_nonempty_requirements,
                task=value["task"],
                conversation=value["conversation"],
                tool_calls=value["tool_calls"],
                domain=value["domain"],
                config=config,
                cache_dir=cache_dir,
            ): key
            for key, value in pending.items()
        }
        for completed_count, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            completed[key] = future.result()
            if completed_count == 1 or completed_count % 10 == 0 or completed_count == len(futures):
                print(
                    json.dumps(
                        {
                            "event": "task_requirement_scoring_progress",
                            "completed": completed_count,
                            "total": len(futures),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    for key, indices in pending_record_indices.items():
        evaluated = completed[key]
        score = evaluated["score"]
        for index in indices:
            records[index]["task_requirements"] = {
                "status": "evaluated",
                "evaluation_key": key,
                **evaluated,
            }
            records[index]["task_requirements_pass"] = score["score"]
            records[index]["task_completion_pass"] = int(score["score"] == 1)

    records.sort(key=lambda row: (row["producer_model"], row["domain"], row["task_id"], row["arm"]))
    domain_prompts = {
        domain: {
            "judge_system_prompt_sha256": hashlib.sha256(
                get_domain_config(domain).judge_system_prompt.encode()
            ).hexdigest(),
            "task_requirements_prompt_sha256": sha256_file(
                get_domain_config(domain).prompts_dir / "judge_task_requirements.md"
            ),
        }
        for domain in DOMAINS
    }
    artifact = {
        "schema": SCHEMA,
        "leaderboard_official": False,
        "protocol_label": PROTOCOL_LABEL,
        "experiment_root": str(experiment_root.resolve()),
        "task_split": task_split,
        "config": asdict(config),
        "scoring_policy": {
            "count_terminal_missing_as_zero": count_terminal_missing_as_zero,
        },
        "judge_prompt_provenance": domain_prompts,
        "inventory": {
            "producer_models": [path.name for path in model_dirs],
            "pairs": len(records) // 2,
            "trajectory_records": len(records),
            "unique_llm_evaluations": len(pending),
            "runtime_failed_shadow_arms": sum(row.path is None for row in inputs),
            "terminal_missing_zero_arms": sum(
                row.path is None
                and row.runtime_error_type
                == "TerminalFailureBeforeBaselinePersistence"
                for row in inputs
            ),
        },
        "metrics": _group_metrics(records),
        "records": records,
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    atomic_json_dump(output_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--producer-model", action="append", dest="producer_models")
    parser.add_argument("--evaluator-model", default="qwen3.8-max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--count-terminal-missing-as-zero",
        action="store_true",
        help=(
            "count tasks marked failed in matrix_status.json before baseline "
            "trajectory persistence as zero in both paired arms"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else output.parent / "_task_requirement_cache" / args.evaluator_model
    )
    artifact = score_experiment(
        experiment_root=args.experiment_root.resolve(),
        output_path=output,
        cache_dir=cache_dir,
        config=ScoreConfig(
            evaluator_model=args.evaluator_model,
            seed=args.seed,
            workers=args.workers,
            max_completion_tokens=args.max_completion_tokens,
            timeout_seconds=args.request_timeout_seconds,
            max_attempts=args.max_attempts,
        ),
        producer_models=tuple(args.producer_models) if args.producer_models else None,
        count_terminal_missing_as_zero=args.count_terminal_missing_as_zero,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "content_sha256": artifact["content_sha256"],
                "pairs": artifact["inventory"]["pairs"],
                "unique_llm_evaluations": artifact["inventory"]["unique_llm_evaluations"],
                "leaderboard_official": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
