"""Side-effect-free reproduction of the official legacy τ-bench reward."""

from __future__ import annotations

from typing import Any


def evaluate(environment: Any, task: Any) -> dict[str, Any]:
    """Score the candidate DB and responses without replacing it with gold DB."""

    candidate_hash = environment.get_data_hash()
    gold_data = environment.data_load_func()
    gold_errors: list[dict[str, Any]] = []
    for index, action in enumerate(task.actions):
        if action.name in environment.terminate_tools:
            continue
        tool = environment.tools_map.get(action.name)
        if tool is None:
            continue
        try:
            observation = tool.invoke(data=gold_data, **action.kwargs)
        except Exception as exc:
            observation = f"Error: {exc}"
        if isinstance(observation, str) and observation.startswith("Error:"):
            gold_errors.append(
                {
                    "action_index": index,
                    "name": action.name,
                    "observation": observation,
                }
            )

    from tau_bench.envs.base import consistent_hash, to_hashable
    from tau_bench.types import RESPOND_ACTION_NAME

    gold_hash = consistent_hash(to_hashable(gold_data))
    state_match = candidate_hash == gold_hash
    output_checks: dict[str, bool] = {}
    for required in task.outputs:
        output_checks[required] = any(
            action.name == RESPOND_ACTION_NAME
            and required.lower()
            in action.kwargs["content"].lower().replace(",", "")
            for action in environment.actions
        )
    output_match = all(output_checks.values())
    return {
        "reward": float(state_match and output_match),
        "state_match": state_match,
        "output_match": output_match,
        "output_checks": output_checks,
        "candidate_data_hash": candidate_hash,
        "gold_data_hash": gold_hash,
        "gold_replay_errors": gold_errors,
        "gold_replay_clean": not gold_errors,
        "score_protocol": "legacy_tau_bench_state_hash_plus_output_substring",
        "candidate_state_preserved_during_scoring": True,
    }
