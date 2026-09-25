"""τ² CandidateExecution schema and projection for policy tests."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from .evidence import project_public_history, project_tool_result
from .model import Tau2ToolBatch


CANDIDATE_EXECUTION_SCHEMA = "candidate-execution-v1"


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _namespaced_schema(
    schema: Mapping[str, Any],
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = copy.deepcopy(dict(schema))
    definitions = raw.pop("$defs", {})

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            rewritten = {key: rewrite(item) for key, item in value.items()}
            ref = rewritten.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                rewritten["$ref"] = (
                    f"#/$defs/{prefix}_{ref.rsplit('/', 1)[-1]}"
                )
            return rewritten
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    renamed = {
        f"{prefix}_{name}": rewrite(value)
        for name, value in definitions.items()
    }
    root_name = f"{prefix}_root"
    renamed[root_name] = rewrite(raw)
    return {"$ref": f"#/$defs/{root_name}"}, renamed


class Tau2CandidateExecutionProvider:
    def __init__(self, environment: Any) -> None:
        domain = environment.get_domain_name()
        if not isinstance(domain, str) or not domain:
            raise ValueError("τ² environment must expose a domain name")
        self.domain_name = domain
        self._schema = self._build_schema(environment, domain)

    @property
    def candidate_schema(self) -> dict[str, Any]:
        return _json_clone(self._schema)

    @staticmethod
    def _db_schema(toolkit: Any) -> Mapping[str, Any] | None:
        if toolkit is None or getattr(toolkit, "db", None) is None:
            return None
        getter = getattr(toolkit.db, "model_json_schema", None)
        if not callable(getter):
            raise TypeError("τ² toolkit DB must expose model_json_schema()")
        schema = getter()
        if not isinstance(schema, Mapping):
            raise TypeError("τ² DB schema must be a mapping")
        return schema

    @classmethod
    def _build_schema(cls, environment: Any, domain: str) -> dict[str, Any]:
        definitions: dict[str, Any] = {}
        agent_raw = cls._db_schema(getattr(environment, "tools", None))
        user_raw = cls._db_schema(getattr(environment, "user_tools", None))
        if agent_raw is None:
            agent_schema: dict[str, Any] = {"type": "null"}
        else:
            agent_schema, agent_defs = _namespaced_schema(agent_raw, "agent_db")
            definitions.update(agent_defs)
        if user_raw is None:
            user_schema: dict[str, Any] = {"type": "null"}
        else:
            user_schema, user_defs = _namespaced_schema(user_raw, "user_db")
            definitions.update(user_defs)
        state_schema = {
            "type": "object",
            "properties": {
                "agent_db": agent_schema,
                "user_db": user_schema,
            },
            "required": ["agent_db", "user_db"],
            "additionalProperties": False,
        }
        call_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "arguments": {"type": "object"},
                "requestor": {"const": "assistant"},
            },
            "required": ["id", "name", "arguments", "requestor"],
            "additionalProperties": False,
        }
        result_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "requestor": {"const": "assistant"},
                "content": {"type": ["string", "null"]},
                "content_is_json": {"type": "boolean"},
                "content_json": {},
                "error": {"type": "boolean"},
            },
            "required": [
                "id",
                "requestor",
                "content",
                "content_is_json",
                "content_json",
                "error",
            ],
            "additionalProperties": False,
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": definitions,
            "type": "object",
            "properties": {
                "schema": {"const": CANDIDATE_EXECUTION_SCHEMA},
                "benchmark": {"const": "tau2"},
                "domain": {"const": domain},
                "now": {"type": "null"},
                "public_history": {"type": "array", "items": {"type": "object"}},
                "action": {
                    "type": "object",
                    "properties": {
                        "calls": {"type": "array", "minItems": 1, "items": call_schema}
                    },
                    "required": ["calls"],
                    "additionalProperties": False,
                },
                "tool_results": {
                    "type": "array",
                    "minItems": 1,
                    "items": result_schema,
                },
                "before_state": state_schema,
                "after_state": state_schema,
            },
            "required": [
                "schema",
                "benchmark",
                "domain",
                "now",
                "public_history",
                "action",
                "tool_results",
                "before_state",
                "after_state",
            ],
            "additionalProperties": False,
        }

    def build_candidate_execution(
        self,
        *,
        public_history: tuple[Any, ...],
        action: Tau2ToolBatch,
        results: tuple[Any, ...],
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        projected_results: list[dict[str, Any]] = []
        for value in results:
            row = project_tool_result(value)
            try:
                content_json = json.loads(row["content"])
                content_is_json = True
            except (TypeError, json.JSONDecodeError):
                content_json = None
                content_is_json = False
            projected_results.append(
                {
                    **row,
                    "content_is_json": content_is_json,
                    "content_json": content_json,
                }
            )
        return {
            "schema": CANDIDATE_EXECUTION_SCHEMA,
            "benchmark": "tau2",
            "domain": self.domain_name,
            "now": None,
            "public_history": project_public_history(public_history),
            "action": {"calls": action.as_json()},
            "tool_results": projected_results,
            "before_state": _json_clone(before_state),
            "after_state": _json_clone(after_state),
        }
