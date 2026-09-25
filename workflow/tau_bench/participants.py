"""Official τ-bench user prompt driven by the repository backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParticipantMetrics:
    model_invocations: int = 0
    api_calls: int = 0
    replay_cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_seconds: float = 0.0

    def observe(self, completion: Any, backend: Any) -> None:
        self.model_invocations += 1
        cache_hit = bool(getattr(backend, "last_cache_hit", False))
        self.replay_cache_hits += int(cache_hit)
        self.api_calls += int(not cache_hit)
        self.prompt_tokens += completion.usage.prompt_tokens
        self.completion_tokens += completion.usage.completion_tokens
        self.total_tokens += completion.usage.total_tokens
        self.generation_seconds += completion.latency_s

    def as_json(self) -> dict[str, Any]:
        return {
            "model_invocations": self.model_invocations,
            "api_calls": self.api_calls,
            "replay_cache_hits": self.replay_cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generation_seconds": round(self.generation_seconds, 6),
        }


class BackendUserSimulator:
    """Behavior-equivalent replacement for the official LiteLLM user class."""

    def __init__(self, *, instruction: str, backend: Any, seed: int) -> None:
        self.instruction = instruction
        self.backend = backend
        self.seed = seed
        self.messages: list[dict[str, Any]] = []
        self.metrics = ParticipantMetrics()

    def _system_prompt(self) -> str:
        return f"""You are a user interacting with an agent.

Instruction: {self.instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def reset(self, instruction: str | None = None) -> str:
        if instruction is not None and instruction != self.instruction:
            raise ValueError("user simulator instruction changed within an episode")
        self.messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self._generate()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._generate()

    def _generate(self) -> str:
        completion = self.backend.complete(
            messages=self.messages,
            tools=None,
            seed=self.seed,
        )
        self.metrics.observe(completion, self.backend)
        if completion.tool_calls or not isinstance(completion.content, str):
            raise RuntimeError("τ-bench user simulator must return text")
        self.messages.append({"role": "assistant", "content": completion.content})
        return completion.content

    def get_total_cost(self) -> float:
        return 0.0
