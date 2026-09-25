"""Optional policy-to-unit-test support for workflow adapters.

The Shadow kernel does not import this package.  Benchmark adapters opt in by
loading a compiled suite, building a JSON ``CandidateExecution``, and appending
the deterministic report to their normal ``Evidence`` value.
"""

from .model import (
    POLICY_TEST_REPORT_SCHEMA,
    POLICY_TEST_SUITE_SCHEMA,
    CompiledPolicyTest,
    PolicyTestSuite,
    canonical_sha256,
    load_policy_test_suite,
    write_policy_test_suite,
)
from .runner import CandidatePolicyTestRunner, PolicyTestExecutionError

__all__ = [
    "POLICY_TEST_REPORT_SCHEMA",
    "POLICY_TEST_SUITE_SCHEMA",
    "CandidatePolicyTestRunner",
    "CompiledPolicyTest",
    "PolicyTestExecutionError",
    "PolicyTestSuite",
    "canonical_sha256",
    "load_policy_test_suite",
    "write_policy_test_suite",
]
