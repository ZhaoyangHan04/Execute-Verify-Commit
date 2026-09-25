"""Print a resolved plan by default; --execute runs user-supplied benchmark tasks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

from .config import ROOT, adapter_config, load_config, resolve


def run_state(plan, domain, task_id, seed, destination):
    from shadow_verifier.backends.dashscope import DashScopeConfig
    from shadow_verifier.backends.hosted import hosted_backend
    from shadow_verifier.experiments import atomic_json_dump
    from workflow.experiment_replay import OccurrenceCacheBackend
    from workflow.state_bench.runner import (
        load_official_task,
        make_generic_reviewer_factory,
    )
    from workflow.state_bench.orchestrator import run_state_bench_task
    from workflow.state_bench.participants import (
        HarnessAgentFactory,
        BackendUserSimulator,
    )
    from workflow.state_bench.local_scoring import (
        ScoreConfig,
        _state_diff,
        _tool_calls,
        _evaluate_nonempty_requirements,
        evaluate_state_requirements,
        evaluate_task_requirements_empty,
    )

    cfg = adapter_config(plan, domain, seed)
    loaded = load_official_task(
        domain_name=domain, task_id=task_id, task_split=cfg.task_split
    )
    g, r, u, outcome = (
        plan["generation"],
        plan["review"],
        plan["simulator"],
        plan["outcome_judge"],
    )

    def backend(model, role):
        temperature = (
            r["temperature"]
            if role == "verifier"
            else u["temperature"]
            if role == "user"
            else g["temperature"]
        )
        cap = (
            r["max_completion_tokens"]
            if role == "verifier"
            else u["max_completion_tokens"]
            if role == "user"
            else g["max_completion_tokens"]
        )
        raw = hosted_backend(
            model=model,
            role=role,
            config=DashScopeConfig(
                temperature=temperature,
                top_p=1.0,
                max_completion_tokens=cap,
                enable_thinking=False,
                timeout_seconds=g["timeout_seconds"],
            ),
        )
        return OccurrenceCacheBackend(raw, destination / "cache" / role)

    agent = HarnessAgentFactory(
        backend=backend(cfg.producer_model, "producer"), seed=seed
    )
    simulator = BackendUserSimulator(
        backend=backend(cfg.user_model, "user"),
        seed=seed,
        system_prompt=loaded.domain.build_simulator_prompt(
            loaded.task, loaded.env_data, loaded.task.user_id
        ),
    )
    factory = None
    if r["mode"] == "shadow":
        reviewer = backend(cfg.judge_model, "verifier")
        factory = make_generic_reviewer_factory(
            backend_factory=lambda _: reviewer,
            model=cfg.judge_model,
            seed=seed,
            variant=cfg.reviewer_variant,
            seed_mode=cfg.reviewer_seed_mode,
            soft_label_output=cfg.reviewer_soft_label_output,
            candidate_effect_visibility=cfg.reviewer_candidate_effect_visibility,
        )
    trajectory = destination / "trajectory.json"
    if not trajectory.exists():
        events = []

        def audit(event):
            events.append(event)
            atomic_json_dump(destination / "review_audit.json", {"records": events})

        run = run_state_bench_task(
            task=loaded.task,
            env_data=loaded.env_data,
            user_id=loaded.task.user_id,
            domain=loaded.domain,
            agent_factory=agent,
            simulator=simulator,
            mode=r["mode"],
            reviewer_factory=factory,
            agent_episode_id="episode-"
            + hashlib.sha256(f"{domain}:{task_id}:{seed}".encode()).hexdigest()[:24],
            verification_budget=cfg.verification_budget,
            probability_aggregation=cfg.reviewer_probability_aggregation,
            probability_threshold=cfg.reviewer_probability_threshold,
            probability_early_stop=cfg.reviewer_probability_early_stop,
            include_tool_schemas_in_evidence=cfg.include_tool_schemas_in_evidence,
            max_tool_rounds=cfg.max_tool_rounds,
            on_review_event=audit,
            trajectory_metadata={"config": asdict(cfg)},
        )
        atomic_json_dump(trajectory, run.trajectory.to_dict())
    record = json.loads(trajectory.read_text())
    state = evaluate_state_requirements(
        loaded.task, _state_diff(record["state_diff"], path=trajectory)
    )
    score = {"state_requirements": state.to_dict(), "task_completion_pass": 0}
    if state.score == 1:
        empty = evaluate_task_requirements_empty(loaded.task)
        requirements = (
            {"score": empty.to_dict()}
            if empty is not None
            else _evaluate_nonempty_requirements(
                task=loaded.task,
                conversation=record["conversation"],
                tool_calls=_tool_calls(record["conversation"]),
                domain=domain,
                config=ScoreConfig(
                    evaluator_model=outcome["model"],
                    seed=seed,
                    max_completion_tokens=outcome["state_max_completion_tokens"],
                    timeout_seconds=g["timeout_seconds"],
                ),
                cache_dir=destination / "scorer_cache",
            )
        )
        score["task_requirements"] = requirements
        score["task_completion_pass"] = int(requirements["score"]["score"] == 1)
    atomic_json_dump(destination / "score.json", score)
    return score


def run_one(plan, domain, task_id, seed, output):
    from shadow_verifier.experiments import atomic_json_dump

    # Content-address the configuration so a changed setting cannot reuse another run.
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    root = output / plan["benchmark"] / plan["setting"] / identity / f"s{seed}"
    atomic_json_dump(root / "configuration.json", plan)
    if plan["benchmark"] == "state_bench":
        slug = hashlib.sha256(f"{domain}:{task_id}".encode()).hexdigest()[:24]
        return run_state(plan, domain, task_id, seed, root / slug)
    cfg = adapter_config(plan, domain, seed)
    if plan["benchmark"] == "tau_bench":
        from workflow.tau_bench.runner import run_task

        return run_task(task_id=int(task_id), config=cfg, output_root=root)
    from workflow.tau2.runner import load_pilot_tasks, run_task

    task = load_pilot_tasks([task_id], domain=domain, task_split=cfg.task_split)[0]
    return run_task(task=task, config=cfg, output_root=root)


def main(argv=None):
    c = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=c["benchmarks"], required=True)
    parser.add_argument("--domain")
    parser.add_argument(
        "--tasks", nargs="+", help="Explicit upstream task IDs; no task data is bundled"
    )
    parser.add_argument(
        "--producer", choices=c["producer_models"], default=c["default_producer"]
    )
    parser.add_argument(
        "--reviewer",
        choices=[*c["producer_models"], "self"],
        default=c["default_reviewer"],
    )
    parser.add_argument(
        "--setting", choices=c["settings"], default=c["default_setting"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=c["seeds"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = resolve(
        args.benchmark, args.setting, args.producer, args.reviewer, args.seeds
    )
    if (
        args.domain is not None
        and args.domain not in plan["benchmark_config"]["domains"]
    ):
        parser.error("Domain is not part of this benchmark")
    if args.workers < 1:
        parser.error("workers must be positive")
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    if not args.domain or not args.tasks:
        parser.error("--execute requires --domain and explicit --tasks")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Task IDs must be unique")
    # Remove proxy inheritance at this command's boundary, never machine-wide.
    for variable in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(variable, None)
    benchmark_root = ROOT / plan["benchmark_config"]["directory"]
    if not benchmark_root.is_dir():
        parser.error("Benchmark missing; see experiments.prepare and the README")
    sys.path.insert(
        0, str(benchmark_root / "src" if args.benchmark == "tau2" else benchmark_root)
    )
    if args.benchmark == "tau2":
        os.environ["TAU2_DATA_DIR"] = str(benchmark_root / "data")
    jobs = [(task, seed) for task in args.tasks for seed in args.seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_one, plan, args.domain, task, seed, args.output)
            for task, seed in jobs
        ]
        for future in futures:
            future.result()  # Errors are explicit, never silently counted as task failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
