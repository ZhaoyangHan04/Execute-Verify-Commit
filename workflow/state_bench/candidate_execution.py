"""STATE-Bench CandidateExecution schema and projection for policy tests."""

from __future__ import annotations

import importlib
import json
import types
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints

from .evidence import project_public_history, project_tool_result
from .model import StateBenchToolBatch


CANDIDATE_EXECUTION_SCHEMA = "candidate-execution-v1"
_SNAPSHOT_RECORDS: dict[str, dict[str, str]] = {
    "travel": {
        "bookings": "Booking",
        "users": "User",
        "hotel_inventory": "HotelInventoryItem",
        "hotels": "HotelReservation",
        "car_inventory": "CarInventoryItem",
        "car_rentals": "CarRental",
    },
    "customer_support": {
        "orders": "Order",
        "order_items": "OrderItem",
        "customers": "Customer",
        "warranties": "Warranty",
    },
    "shopping_assistant": {
        "carts": "Cart",
        "cart_items": "CartItem",
        "customers": "Customer",
    },
}


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


class _DataclassSchemaBuilder:
    def __init__(self) -> None:
        self.definitions: dict[str, Any] = {}
        self._building: set[str] = set()

    def schema(self, annotation: Any) -> dict[str, Any]:
        if annotation is Any:
            return {}
        if annotation is None or annotation is type(None):
            return {"type": "null"}
        if annotation in (str, int, float, bool):
            return {
                "type": {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                }[annotation]
            }
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Literal:
            values = list(args)
            return {"enum": values}
        if origin in (Union, types.UnionType):
            return {"anyOf": [self.schema(value) for value in args]}
        if origin in (list, tuple, set, frozenset):
            item = args[0] if args else Any
            return {"type": "array", "items": self.schema(item)}
        if origin is dict:
            value = args[1] if len(args) == 2 else Any
            return {
                "type": "object",
                "additionalProperties": self.schema(value),
            }
        if is_dataclass(annotation):
            name = annotation.__name__
            if name not in self.definitions and name not in self._building:
                self._building.add(name)
                hints = get_type_hints(annotation)
                properties = {
                    field.name: self.schema(hints.get(field.name, Any))
                    for field in fields(annotation)
                }
                self.definitions[name] = {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                }
                self._building.remove(name)
            return {"$ref": f"#/$defs/{name}"}
        return {}


def state_bench_policy_source(domain_name: str) -> str:
    if domain_name not in _SNAPSHOT_RECORDS:
        raise ValueError(f"unsupported STATE-Bench domain: {domain_name!r}")
    module_names = [f"state_bench.domains.{domain_name}.policies"]
    if domain_name == "travel":
        module_names.append("state_bench.domains.travel.policy_texts")
    sections: list[str] = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        path = Path(module.__file__ or "")
        sections.append(
            f"# SOURCE {module_name}\n" + path.read_text(encoding="utf-8")
        )
    return "\n\n".join(sections)


class StateBenchCandidateExecutionProvider:
    def __init__(self, domain_name: str) -> None:
        if domain_name not in _SNAPSHOT_RECORDS:
            raise ValueError(f"unsupported STATE-Bench domain: {domain_name!r}")
        self.domain_name = domain_name
        self._schema = self._build_schema(domain_name)

    @property
    def candidate_schema(self) -> dict[str, Any]:
        return _json_clone(self._schema)

    @staticmethod
    def _build_schema(domain_name: str) -> dict[str, Any]:
        schemas = importlib.import_module(
            f"state_bench.domains.{domain_name}.schemas"
        )
        builder = _DataclassSchemaBuilder()
        snapshot_properties: dict[str, Any] = {}
        for table, class_name in _SNAPSHOT_RECORDS[domain_name].items():
            record_type = getattr(schemas, class_name)
            snapshot_properties[table] = {
                "type": "object",
                "additionalProperties": builder.schema(record_type),
            }
        state_schema = {
            "type": "object",
            "properties": snapshot_properties,
            "required": list(snapshot_properties),
            "additionalProperties": False,
        }
        call_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["id", "name", "arguments"],
            "additionalProperties": False,
        }
        result_schema = {
            "type": "object",
            "properties": {**call_schema["properties"], "result": {}},
            "required": ["id", "name", "arguments", "result"],
            "additionalProperties": False,
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": builder.definitions,
            "type": "object",
            "properties": {
                "schema": {"const": CANDIDATE_EXECUTION_SCHEMA},
                "benchmark": {"const": "state_bench"},
                "domain": {"const": domain_name},
                "now": {"type": "string"},
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
        now: str,
        public_history: tuple[Any, ...],
        action: StateBenchToolBatch,
        observations: tuple[Mapping[str, Any], ...],
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(now, str) or not now:
            raise ValueError("STATE-Bench CandidateExecution now must be non-empty")
        return {
            "schema": CANDIDATE_EXECUTION_SCHEMA,
            "benchmark": "state_bench",
            "domain": self.domain_name,
            "now": now,
            "public_history": project_public_history(public_history),
            "action": {"calls": action.as_json()},
            "tool_results": [project_tool_result(value) for value in observations],
            "before_state": _json_clone(before_state),
            "after_state": _json_clone(after_state),
        }
