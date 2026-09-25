"""Resolve paper settings before importing any benchmark or contacting a model."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "paper.json"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = Path(__file__).with_name("paper.json")


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def resolve(benchmark, setting=None, producer=None, reviewer=None, seeds=None):
    cfg = load_config()
    setting = setting or cfg["default_setting"]
    producer = producer or cfg["default_producer"]
    reviewer = producer if reviewer == "self" else reviewer or cfg["default_reviewer"]
    if benchmark not in cfg["benchmarks"] or setting not in cfg["settings"]:
        raise ValueError("Unknown benchmark or setting")
    if producer not in cfg["producer_models"] or reviewer not in cfg["producer_models"]:
        raise ValueError("This entry point exposes only the paper's generative models")
    spec = cfg["settings"][setting]
    if setting in {"b3", "b5", "b7", "prob_es", "hidden_effect"} and (
        producer != "qwen3.8-max" or reviewer != "qwen3.8-max"
    ):
        raise ValueError("The paper's mechanism/compute settings use Qwen self-review")
    seeds = cfg["seeds"] if seeds is None else list(seeds)
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(s not in cfg["seeds"] for s in seeds)
    ):
        raise ValueError("Choose unique seeds from 42, 43, 44, 45")
    return {
        "benchmark": benchmark,
        "setting": setting,
        "producer": producer,
        "reviewer": spec.get("reviewer", reviewer),
        "seeds": seeds,
        "generation": cfg["generation"],
        "simulator": cfg["simulator"],
        "outcome_judge": cfg["outcome_judge"],
        "benchmark_config": cfg["benchmarks"][benchmark],
        "review": {**cfg["review"], **spec},
    }


def review_kwargs(plan):
    r = plan["review"]
    return dict(
        verification_budget=r["budget"],
        reviewer_temperature=r["temperature"],
        reviewer_variant=r["prompt"],
        reviewer_seed_mode=r["seed_mode"],
        reviewer_soft_label_output=r.get("soft_label_output", False),
        reviewer_probability_aggregation=r.get(
            "probability_aggregation", "vote_majority"
        ),
        reviewer_probability_threshold=r.get("probability_threshold", 0.5),
        reviewer_probability_early_stop=r.get("probability_early_stop", False),
        include_tool_schemas_in_evidence=r["include_tool_list"],
        reviewer_candidate_effect_visibility=r["candidate_effect_visibility"],
    )


def adapter_config(plan, domain, seed):
    b, g, outcome = plan["benchmark_config"], plan["generation"], plan["outcome_judge"]
    values = dict(
        producer_model=plan["producer"],
        judge_model=plan["reviewer"],
        user_model=plan["simulator"]["runtime_models"][plan["benchmark"]],
        seed=seed,
        **review_kwargs(plan),
    )
    if plan["benchmark"] == "state_bench":
        from workflow.state_bench.runner import StateBenchPairConfig

        return StateBenchPairConfig(
            **values, max_tool_rounds=b["max_tool_rounds"], task_split=b["split"]
        )
    values.update(
        mode=plan["review"]["mode"],
        domain=domain,
        temperature=g["temperature"],
        top_p=g["top_p"],
        max_completion_tokens=g["max_completion_tokens"],
        judge_max_completion_tokens=plan["review"]["max_completion_tokens"],
        max_steps=b["max_steps"],
        request_timeout_seconds=g["timeout_seconds"],
        pairing_mode="independent",
    )
    if plan["benchmark"] == "tau_bench":
        from workflow.tau_bench.runner import TauBenchConfig

        return TauBenchConfig(**values)
    from workflow.tau2.runner import Tau2PilotConfig

    return Tau2PilotConfig(
        **values,
        task_split=b["split"],
        max_errors=b["max_errors"],
        episode_timeout_seconds=b["episode_timeout_seconds"],
        enable_thinking=g["enable_thinking"],
        nl_evaluator_model=outcome["model"],
        nl_evaluator_temperature=outcome["temperature"],
        nl_evaluator_top_p=outcome["top_p"],
        nl_evaluator_enable_thinking=outcome["enable_thinking"],
        nl_evaluator_max_completion_tokens=outcome["tau2_max_completion_tokens"],
        nl_evaluator_seed=outcome["tau2_seed"],
    )
