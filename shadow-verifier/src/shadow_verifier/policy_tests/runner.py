"""Deterministic execution of a compiled policy-test suite."""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .model import (
    POLICY_TEST_REPORT_SCHEMA,
    CompiledPolicyTest,
    PolicyTestSuite,
    canonical_sha256,
)


class PolicyTestExecutionError(RuntimeError):
    """The suite or CandidateExecution violates the deterministic contract."""


_HELPER_NAMES = frozenset(
    {
        "read",
        "len",
        "sum",
        "min",
        "max",
        "all",
        "any",
        "abs",
        "round",
        "int",
        "float",
        "str",
        "bool",
        "sorted",
        "enumerate",
        "zip",
        "range",
        "list",
        "tuple",
        "dict",
        "set",
        "isinstance",
        "next",
    }
)
_SAFE_JSON_METHODS = frozenset(
    {
        "get",
        "items",
        "keys",
        "values",
        "startswith",
        "endswith",
        "lower",
        "upper",
        "strip",
        "split",
        "append",
    }
)
_FORBIDDEN_NODES = (
    ast.Attribute,
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.Raise,
    ast.With,
    ast.AsyncWith,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.NamedExpr,
)
_PROTECTED_NAMES = _HELPER_NAMES | frozenset({"__builtins__"})


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.nodes += 1
        if self.nodes > 1_500:
            raise ValueError("unit-test code has too many AST nodes")
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"forbidden syntax in unit-test code: {type(node).__name__}")
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Pow, ast.LShift, ast.RShift, ast.MatMult)
        ):
            raise ValueError("forbidden arithmetic operator in unit-test code")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private names are forbidden in unit-test code")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 4_000:
                raise ValueError("oversized string in unit-test code")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _HELPER_NAMES:
            self.generic_visit(node)
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr in _SAFE_JSON_METHODS:
            self.visit(node.func.value)
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        if isinstance(node.func, ast.Name):
            called = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr
        else:
            called = type(node.func).__name__
        raise ValueError(
            f"unit-test code called non-allowlisted function or method: {called}"
        )

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in _PROTECTED_NAMES:
            raise ValueError("unit-test code overwrote a protected helper")
        self.generic_visit(node)


def _compile_body(test: CompiledPolicyTest) -> Any:
    try:
        tree = ast.parse(test.code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"invalid Python in {test.test_id}") from exc
    _SafetyVisitor().visit(tree)
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    missing = {"applicable", "passed", "observed"} - assigned
    if missing:
        raise ValueError(
            f"{test.test_id} does not assign required outputs: {sorted(missing)!r}"
        )
    return compile(tree, f"<policy-test:{test.test_id}>", "exec")


def _schema_at_path(schema: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    root = schema
    current: Mapping[str, Any] = root
    for component in path.split("."):
        while "$ref" in current:
            ref = current["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
                raise ValueError("only local $defs references are supported")
            current = root["$defs"][ref.rsplit("/", 1)[-1]]
        alternatives = current.get("anyOf")
        if isinstance(alternatives, list):
            non_null = [
                value
                for value in alternatives
                if isinstance(value, Mapping) and value.get("type") != "null"
            ]
            if len(non_null) == 1:
                current = non_null[0]
                while "$ref" in current:
                    ref = current["$ref"]
                    current = root["$defs"][ref.rsplit("/", 1)[-1]]
        properties = current.get("properties")
        if not isinstance(properties, Mapping) or component not in properties:
            raise ValueError(f"declared read path is absent from schema: {path}")
        child = properties[component]
        if not isinstance(child, Mapping):
            raise ValueError(f"invalid schema node for read path: {path}")
        current = child
    return current


def _bounded_range(*args: int) -> range:
    value = range(*args)
    if len(value) > 10_000:
        raise ValueError("unit-test range exceeds 10,000 iterations")
    return value


def _json_observed(value: Any) -> Any:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > 12_000:
        raise ValueError("unit-test observed output exceeds 12KB")
    return json.loads(encoded)


@dataclass(frozen=True)
class _ExecutableTest:
    definition: CompiledPolicyTest
    code: Any


class CandidatePolicyTestRunner:
    """Validate one CandidateExecution and run a content-addressed suite."""

    def __init__(
        self,
        suite: PolicyTestSuite,
        candidate_schema: Mapping[str, Any],
    ) -> None:
        if not isinstance(suite, PolicyTestSuite):
            raise TypeError("suite must be PolicyTestSuite")
        if not isinstance(candidate_schema, Mapping):
            raise TypeError("candidate_schema must be a mapping")
        if canonical_sha256(candidate_schema) != suite.candidate_schema_sha256:
            raise ValueError("suite and CandidateExecution schema hashes differ")
        Draft202012Validator.check_schema(candidate_schema)
        self._validator = Draft202012Validator(candidate_schema)
        self._suite = suite
        executable: list[_ExecutableTest] = []
        for test in suite.tests:
            for path in test.reads:
                _schema_at_path(candidate_schema, path)
            executable.append(_ExecutableTest(test, _compile_body(test)))
        self._tests = tuple(executable)

    @property
    def suite(self) -> PolicyTestSuite:
        return self._suite

    def run(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        try:
            detached = json.loads(
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise PolicyTestExecutionError("CandidateExecution is not finite JSON") from exc
        errors = sorted(self._validator.iter_errors(detached), key=lambda item: list(item.path))
        if errors:
            first = errors[0]
            location = ".".join(str(value) for value in first.absolute_path) or "<root>"
            raise PolicyTestExecutionError(
                f"CandidateExecution schema validation failed at {location}"
            )
        calls = detached["action"]["calls"]
        action_names = {
            call["name"] for call in calls if isinstance(call, dict) and "name" in call
        }
        rows = [self._run_one(test, detached, action_names) for test in self._tests]
        counts = {
            status: sum(row["status"] == status for row in rows)
            for status in ("passed", "failed", "not_applicable", "error")
        }
        return {
            "schema": POLICY_TEST_REPORT_SCHEMA,
            "suite_sha256": self._suite.suite_sha256,
            "compiler_model": self._suite.compiler_model,
            "candidate_schema_sha256": self._suite.candidate_schema_sha256,
            "summary": counts,
            "tests": rows,
        }

    @staticmethod
    def _run_one(
        executable: _ExecutableTest,
        candidate: Mapping[str, Any],
        action_names: set[str],
    ) -> dict[str, Any]:
        test = executable.definition
        base = {
            "test_id": test.test_id,
            "meaning": test.meaning,
            "policy_basis": test.policy_basis,
        }
        if not action_names.intersection(test.applies_to):
            return {**base, "status": "not_applicable", "observed": {}}

        declared = frozenset(test.reads)

        def read(path: str) -> Any:
            if path not in declared:
                raise ValueError("unit test attempted an undeclared read")
            value: Any = candidate
            for component in path.split("."):
                if not isinstance(value, Mapping) or component not in value:
                    raise ValueError("declared CandidateExecution path is missing")
                value = value[component]
            return copy.deepcopy(value)

        namespace: dict[str, Any] = {
            "__builtins__": {},
            "read": read,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "all": all,
            "any": any,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "range": _bounded_range,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "set": set,
            "isinstance": isinstance,
            "next": next,
        }
        namespace.update(
            {
                "applicable": True,
                "passed": False,
                "observed": {},
            }
        )
        try:
            exec(executable.code, namespace, namespace)
            applicable = namespace.get("applicable")
            passed = namespace.get("passed")
            if type(applicable) is not bool or type(passed) is not bool:
                raise TypeError("applicable and passed must be exactly bool")
            observed = _json_observed(namespace.get("observed"))
        except Exception as exc:
            return {
                **base,
                "status": "error",
                "observed": {"error_type": type(exc).__name__},
            }
        status = "not_applicable" if not applicable else "passed" if passed else "failed"
        return {**base, "status": status, "observed": observed}
