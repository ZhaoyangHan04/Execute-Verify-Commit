"""Strict, content-addressed values for compiled policy unit tests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


POLICY_TEST_SUITE_SCHEMA = "candidate-policy-test-suite-v1"
POLICY_TEST_REPORT_SCHEMA = "candidate-policy-test-report-v1"
_TEST_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PATH = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class CompiledPolicyTest:
    test_id: str
    meaning: str
    policy_basis: str
    applies_to: tuple[str, ...]
    reads: tuple[str, ...]
    code: str

    def __post_init__(self) -> None:
        if not isinstance(self.test_id, str) or _TEST_ID.fullmatch(self.test_id) is None:
            raise ValueError("test_id must be a lowercase snake_case token")
        for name in ("meaning", "policy_basis", "code"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if len(self.meaning) > 500 or len(self.policy_basis) > 1_000:
            raise ValueError("unit-test description is unreasonably long")
        if len(self.code.encode("utf-8")) > 12_000:
            raise ValueError("unit-test code exceeds 12KB")
        if (
            not isinstance(self.applies_to, tuple)
            or not self.applies_to
            or len(set(self.applies_to)) != len(self.applies_to)
            or not all(isinstance(value, str) and value for value in self.applies_to)
        ):
            raise ValueError("applies_to must be a unique non-empty tool-name tuple")
        if (
            not isinstance(self.reads, tuple)
            or not self.reads
            or len(set(self.reads)) != len(self.reads)
            or not all(isinstance(value, str) and _PATH.fullmatch(value) for value in self.reads)
        ):
            raise ValueError("reads must be a unique non-empty dotted-path tuple")


@dataclass(frozen=True)
class PolicyTestSuite:
    benchmark: str
    domain: str
    compiler_model: str
    seed: int
    policy_sha256: str
    candidate_schema_sha256: str
    tool_schemas_sha256: str
    compiler_prompt_sha256: str
    compiler_completion_sha256: str
    tests: tuple[CompiledPolicyTest, ...]
    compiler_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("benchmark", "domain", "compiler_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if type(self.seed) is not int:
            raise TypeError("seed must be int")
        for name in (
            "policy_sha256",
            "candidate_schema_sha256",
            "tool_schemas_sha256",
            "compiler_prompt_sha256",
            "compiler_completion_sha256",
        ):
            if not isinstance(getattr(self, name), str) or _SHA256.fullmatch(
                getattr(self, name)
            ) is None:
                raise ValueError(f"{name} must be lowercase SHA-256")
        if (
            not isinstance(self.tests, tuple)
            or not self.tests
            or len({test.test_id for test in self.tests}) != len(self.tests)
            or not all(isinstance(test, CompiledPolicyTest) for test in self.tests)
        ):
            raise ValueError("tests must be a unique non-empty CompiledPolicyTest tuple")
        if not isinstance(self.compiler_audit, Mapping):
            raise TypeError("compiler_audit must be a mapping")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_TEST_SUITE_SCHEMA,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "compiler_model": self.compiler_model,
            "seed": self.seed,
            "policy_sha256": self.policy_sha256,
            "candidate_schema_sha256": self.candidate_schema_sha256,
            "tool_schemas_sha256": self.tool_schemas_sha256,
            "compiler_prompt_sha256": self.compiler_prompt_sha256,
            "compiler_completion_sha256": self.compiler_completion_sha256,
            "tests": [asdict(test) for test in self.tests],
            "compiler_audit": dict(self.compiler_audit),
        }

    @property
    def suite_sha256(self) -> str:
        return canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "suite_sha256": self.suite_sha256}


def load_policy_test_suite(path: Path) -> PolicyTestSuite:
    if not isinstance(path, Path):
        raise TypeError("suite path must be pathlib.Path")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read policy-test suite as JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema") != POLICY_TEST_SUITE_SCHEMA:
        raise ValueError("unsupported policy-test suite schema")
    claimed = raw.pop("suite_sha256", None)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise ValueError("suite_sha256 is missing or invalid")
    if canonical_sha256(raw) != claimed:
        raise ValueError("policy-test suite content hash mismatch")
    try:
        tests = tuple(
            CompiledPolicyTest(
                test_id=row["test_id"],
                meaning=row["meaning"],
                policy_basis=row["policy_basis"],
                applies_to=tuple(row["applies_to"]),
                reads=tuple(row["reads"]),
                code=row["code"],
            )
            for row in raw["tests"]
        )
        suite = PolicyTestSuite(
            benchmark=raw["benchmark"],
            domain=raw["domain"],
            compiler_model=raw["compiler_model"],
            seed=raw["seed"],
            policy_sha256=raw["policy_sha256"],
            candidate_schema_sha256=raw["candidate_schema_sha256"],
            tool_schemas_sha256=raw["tool_schemas_sha256"],
            compiler_prompt_sha256=raw["compiler_prompt_sha256"],
            compiler_completion_sha256=raw["compiler_completion_sha256"],
            tests=tests,
            compiler_audit=raw["compiler_audit"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy-test suite is malformed") from exc
    if suite.suite_sha256 != claimed:
        raise ValueError("policy-test suite did not round-trip canonically")
    return suite


def write_policy_test_suite(path: Path, suite: PolicyTestSuite) -> None:
    if not isinstance(path, Path):
        raise TypeError("suite path must be pathlib.Path")
    if not isinstance(suite, PolicyTestSuite):
        raise TypeError("suite must be PolicyTestSuite")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            suite.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
