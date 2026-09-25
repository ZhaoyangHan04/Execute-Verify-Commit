"""Synthetic, offline tests; no benchmark records and no model API calls."""

import json
from types import SimpleNamespace
import unittest

from experiments.config import load_config, resolve, adapter_config
from shadow_verifier import (
    Candidate,
    Decision,
    Evidence,
    Part,
    Published,
    Rejected,
    ShadowVerifier,
)
from shadow_verifier.backends import Completion, CompletionProvenance, Usage
from shadow_verifier.backends.dashscope import DashScopeBackend, DashScopeConfig
from shadow_verifier.reviewers import SemanticReviewerConfig, SemanticReviewerFactory
from shadow_verifier.reviewers.semantic import SemanticReviewer
from shadow_verifier.backends.typesafe import request_payload


def evidence():
    return Evidence(
        context=(Part.text("policy", "Only update the requested record."),),
        action=(Part.json("action", {"name": "set_value", "value": 2}),),
        effect=(Part.json("candidate_result", {"value": 2}),),
    )


class Executor:
    def __init__(self):
        self.value = 1
        self.executions = 0
        self.discards = 0

    def fingerprint(self, action):
        return str(action)

    def stage(self, call_id, action):
        self.executions += 1
        return Candidate(
            call_id=call_id,
            candidate_id="candidate-" + call_id,
            handle=action,
            evidence=evidence(),
        )

    def publish(self, candidate):
        self.value = candidate.handle
        return self.value

    def rejection(self, candidate, decision):
        return decision.rationale

    def discard(self, candidate):
        self.discards += 1


class Votes:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def __call__(self):
        return self.for_sample(0)

    def for_sample(self, index):
        def review(_):
            self.calls += 1
            p = self.probabilities[index]
            return Decision(
                p > 0.5, "accepted" if p > 0.5 else "goal_mismatch", f"vote {index}", p
            )

        return review


class Backend:
    model = "qwen3.8-max"
    config = DashScopeConfig(max_completion_tokens=512)

    def __init__(self):
        self.requests = []

    def complete(self, messages, tools=None, seed=None, json_mode=False):
        self.requests.append((messages, seed))
        text = json.dumps(
            {"accept": True, "code": "accepted", "rationale": "Valid update."}
        )
        return Completion(
            text,
            (),
            "stop",
            self.model,
            Usage(),
            0.0,
            text,
            CompletionProvenance(
                self.model, self.model, "json", False, 0, 0, seed, 0.0, 1.0, 512, False
            ),
        )


class FrameworkTests(unittest.TestCase):
    def test_reject_does_not_mutate_canonical_state(self):
        executor = Executor()
        result = ShadowVerifier(executor, Votes([0.1])).step("one", 2)
        self.assertIsInstance(result, Rejected)
        self.assertEqual(
            (executor.value, executor.executions, executor.discards), (1, 1, 1)
        )

    def test_accept_promotes_once_and_is_idempotent(self):
        executor = Executor()
        gate = ShadowVerifier(executor, Votes([0.9]))
        first = gate.step("one", 2)
        self.assertIsInstance(first, Published)
        self.assertEqual(gate.step("one", 2), first)
        self.assertEqual((executor.value, executor.executions), (2, 1))

    def test_hard_majority_preserves_negative_reasons(self):
        executor = Executor()
        votes = Votes([0.1, 0.9, 0.2])
        result = ShadowVerifier(executor, votes, verification_budget=3).step("one", 2)
        self.assertIsInstance(result, Rejected)
        self.assertIn("vote 0", result.observation)
        self.assertIn("vote 2", result.observation)
        self.assertEqual(votes.calls, 3)

    def test_probability_stops_on_first_strong_vote(self):
        votes = Votes([0.6, 0.2, 0.9, 0.9, 0.9])
        gate = ShadowVerifier(
            Executor(),
            votes,
            verification_budget=5,
            probability_aggregation="mean_probability",
            probability_threshold=0.7,
            probability_early_stop=True,
        )
        self.assertIsInstance(gate.step("one", 2), Rejected)
        self.assertEqual(votes.calls, 2)

    def test_probability_thresholds_are_strict(self):
        votes = Votes([0.3, 0.7, 0.7, 0.7, 0.7])
        gate = ShadowVerifier(
            Executor(),
            votes,
            verification_budget=5,
            probability_aggregation="mean_probability",
            probability_threshold=0.7,
            probability_early_stop=True,
        )
        self.assertIsInstance(gate.step("one", 2), Rejected)
        self.assertEqual(votes.calls, 5)

    def test_hidden_effect_does_not_change_original_evidence(self):
        backend, data = Backend(), evidence()
        reviewer = SemanticReviewer(
            backend,
            backend.model,
            SemanticReviewerConfig(candidate_effect_visibility="hidden"),
        )
        reviewer(data)
        self.assertNotIn("candidate_result", backend.requests[0][0][1]["content"])
        self.assertIn("Only update", backend.requests[0][0][1]["content"])
        self.assertEqual(len(data.effect), 1)

    def test_jev_receives_approval_and_category_questions(self):
        backend = Backend()
        SemanticReviewer(backend, backend.model)(evidence())
        payload = request_payload(backend.requests[0][0], "jev-1.13.0")
        self.assertEqual(set(payload["questions"]), {"approve", "rejection_category"})
        self.assertEqual(payload["questions"]["approve"]["type"], "noul")

    def test_same_seed_panel_uses_fresh_reviewers(self):
        backend = Backend()
        factory = SemanticReviewerFactory(
            lambda _: backend,
            model=backend.model,
            config=SemanticReviewerConfig(seed=42),
        )
        for i in range(5):
            factory.for_sample(i)(evidence())
        self.assertEqual([seed for _, seed in backend.requests], [42] * 5)

    def test_request_caps_and_kimi_native_sampling(self):
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
                self.request = None

            def create(self, **request):
                self.request = request
                return SimpleNamespace(
                    model=request["model"],
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content="ok", tool_calls=None),
                        )
                    ],
                )

        for model in ["qwen3.8-max", "glm-5.2", "kimi-k3"]:
            client = Client()
            backend = DashScopeBackend(
                model, DashScopeConfig(max_completion_tokens=512), client=client
            )
            backend.complete([{"role": "user", "content": "synthetic"}], seed=42)
            self.assertEqual(client.request["max_completion_tokens"], 512)
            for key in ["temperature", "top_p", "seed"]:
                self.assertEqual(key in client.request, model != "kimi-k3")

    def test_paper_configuration(self):
        cfg = load_config()
        self.assertEqual(cfg["seeds"], [42, 43, 44, 45])
        for benchmark, steps in [("tau_bench", 30), ("tau2", 200)]:
            for setting, budget, temperature in [
                ("b1", 1, 0.0),
                ("b3", 3, 0.3),
                ("b5", 5, 0.3),
                ("b7", 7, 0.3),
                ("prob_es", 5, 0.0),
            ]:
                plan = resolve(benchmark, setting)
                c = adapter_config(plan, plan["benchmark_config"]["domains"][0], 42)
                self.assertEqual(
                    (c.verification_budget, c.reviewer_temperature, c.max_steps),
                    (budget, temperature, steps),
                )
                self.assertEqual(
                    (c.max_completion_tokens, c.judge_max_completion_tokens),
                    (8192, 512),
                )
                self.assertEqual(c.reviewer_variant, "plain_checks_v1")
                self.assertEqual(c.reviewer_seed_mode, "fixed")
                self.assertTrue(c.include_tool_schemas_in_evidence)
        self.assertEqual(cfg["benchmarks"]["state_bench"]["max_tool_rounds"], 16)

    def test_jev_never_replaces_outcome_judge(self):
        for benchmark in load_config()["benchmarks"]:
            plan = resolve(benchmark, "jev", "glm-5.2")
            self.assertEqual(plan["reviewer"], "jev-1.13.0")
            self.assertEqual(plan["outcome_judge"]["model"], "qwen3.8-max")


if __name__ == "__main__":
    unittest.main()
