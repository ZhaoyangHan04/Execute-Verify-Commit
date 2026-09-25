"""Static/deterministic audit of the pinned legacy τ-bench test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shadow_verifier.experiments import atomic_json_dump

from .runner import (
    MUTATING_TOOLS,
    READ_ONLY_TOOLS,
    SUPPORTED_DOMAINS,
    _make_environment,
    load_tasks,
)


def audit_domain(domain: str) -> dict[str, Any]:
    tasks = load_tasks(domain)
    task_rows: list[dict[str, Any]] = []
    known_tools: set[str] = set()
    cabin_mismatches: list[dict[str, Any]] = []
    exchange_pair_reorders: list[dict[str, Any]] = []
    for task_id, task in enumerate(tasks):
        environment = _make_environment(domain, task_id)
        errors: list[dict[str, Any]] = []
        for action_index, action in enumerate(task.actions):
            tool = environment.tools_map.get(action.name)
            if tool is None:
                continue
            known_tools.update(environment.tools_map)
            try:
                observation = tool.invoke(data=environment.data, **action.kwargs)
            except Exception as exc:
                observation = f"Error: {exc}"
            if isinstance(observation, str) and observation.startswith("Error:"):
                errors.append(
                    {
                        "action_index": action_index,
                        "name": action.name,
                        "observation": observation,
                        "mutation_capable": action.name in MUTATING_TOOLS[domain],
                    }
                )
            if action.name == "update_reservation_flights" and not str(
                observation
            ).startswith("Error:"):
                reservation = environment.data["reservations"][
                    action.kwargs["reservation_id"]
                ]
                if reservation["cabin"] != action.kwargs["cabin"]:
                    cabin_mismatches.append(
                        {
                            "task_id": task_id,
                            "action_index": action_index,
                            "requested": action.kwargs["cabin"],
                            "observed": reservation["cabin"],
                        }
                    )
            if (
                action.name == "exchange_delivered_order_items"
                and len(action.kwargs["item_ids"]) > 1
                and not str(observation).startswith("Error:")
            ):
                requested = list(
                    zip(action.kwargs["item_ids"], action.kwargs["new_item_ids"])
                )
                order = environment.data["orders"][action.kwargs["order_id"]]
                stored = list(
                    zip(order["exchange_items"], order["exchange_new_items"])
                )
                if requested != stored:
                    exchange_pair_reorders.append(
                        {
                            "task_id": task_id,
                            "action_index": action_index,
                            "requested_pairs": requested,
                            "stored_parallel_lists_zip": stored,
                            "note": (
                                "legacy tool sorts the two set-valued fields "
                                "independently; action arguments remain authoritative"
                            ),
                        }
                    )
        task_rows.append(
            {
                "task_id": task_id,
                "gold_replay_clean": not errors,
                "gold_replay_errors": errors,
            }
        )
    return {
        "schema": "tau-bench-test-split-audit-v1",
        "domain": domain,
        "task_count": len(tasks),
        "gold_replay_clean_tasks": sum(row["gold_replay_clean"] for row in task_rows),
        "gold_replay_error_tasks": sum(not row["gold_replay_clean"] for row in task_rows),
        "gold_replay_error_actions": sum(
            len(row["gold_replay_errors"]) for row in task_rows
        ),
        "gold_mutation_error_actions": sum(
            error["mutation_capable"]
            for row in task_rows
            for error in row["gold_replay_errors"]
        ),
        "unclassified_tools": sorted(
            known_tools - MUTATING_TOOLS[domain] - READ_ONLY_TOOLS[domain]
        ),
        "cabin_mismatches_after_repair": cabin_mismatches,
        "legacy_exchange_parallel_list_reorders": exchange_pair_reorders,
        "tasks": task_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=SUPPORTED_DOMAINS, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    domains = args.domain or list(SUPPORTED_DOMAINS)
    payload = {
        "schema": "tau-bench-test-split-audit-bundle-v1",
        "domains": {domain: audit_domain(domain) for domain in domains},
    }
    if args.output:
        atomic_json_dump(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return int(
        any(
            row["unclassified_tools"] or row["cabin_mismatches_after_repair"]
            for row in payload["domains"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
