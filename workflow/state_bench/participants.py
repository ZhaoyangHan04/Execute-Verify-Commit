"""Provider-neutral STATE-Bench participants for harness-owned execution.

STATE-Bench publishes Responses-style function schemas and a compact
conversation representation.  The completion backends used by the Shadow
kernel consume Chat Completions-style messages instead.  This module is the
only translation boundary between those two protocols.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from shadow_verifier.backends import Completion
from state_bench.agents.base import (
    AgentRuntimeContext,
    AgentToolCallRequest,
    AgentTurnResponse,
    BaseAgent,
)


def _json_clone(value: Any, *, subject: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        raise TypeError(f"{subject} must be finite JSON") from None


def _json_clone_preserving_order(value: Any, *, subject: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        raise TypeError(f"{subject} must be finite JSON") from None


def responses_tool_schema_to_chat(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one pinned STATE-Bench function schema without guessing fields."""

    if not isinstance(schema, Mapping):
        raise TypeError("STATE-Bench tool schema must be a mapping")
    expected = {"type", "name", "description", "parameters"}
    if set(schema) != expected:
        raise ValueError(
            "STATE-Bench tool schema must contain exactly "
            "type, name, description, and parameters"
        )
    if schema["type"] != "function":
        raise ValueError("only function tool schemas are supported")
    name = schema["name"]
    description = schema["description"]
    parameters = schema["parameters"]
    if not isinstance(name, str) or not name:
        raise ValueError("tool schema name must be a non-empty string")
    if not isinstance(description, str):
        raise TypeError("tool schema description must be a string")
    if not isinstance(parameters, Mapping):
        raise TypeError("tool schema parameters must be a mapping")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _json_clone(parameters, subject=f"tool {name!r} parameters"),
        },
    }


def responses_tool_schemas_to_chat(
    schemas: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert and validate the complete official schema registry."""

    if not isinstance(schemas, (list, tuple)):
        raise TypeError("STATE-Bench tool schemas must be a list or tuple")
    converted = [responses_tool_schema_to_chat(schema) for schema in schemas]
    names = [tool["function"]["name"] for tool in converted]
    if len(set(names)) != len(names):
        raise ValueError("STATE-Bench tool schema names must be unique")
    return converted


def _official_call_records(
    value: Any,
    *,
    subject: str,
    preserve_argument_order: bool = False,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{subject} must be a list or null")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{subject}[{index}] must be a mapping")
        if set(raw) != {"name", "arguments", "result"}:
            raise ValueError(
                f"{subject}[{index}] must contain exactly name, arguments, and result"
            )
        name = raw["name"]
        arguments = raw["arguments"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{subject}[{index}].name must be non-empty")
        if not isinstance(arguments, Mapping):
            raise TypeError(f"{subject}[{index}].arguments must be a mapping")
        clone_arguments = (
            _json_clone_preserving_order if preserve_argument_order else _json_clone
        )
        records.append(
            {
                "name": name,
                "arguments": clone_arguments(
                    arguments,
                    subject=f"{subject}[{index}].arguments",
                ),
                "result": _json_clone(raw["result"], subject=f"{subject}[{index}].result"),
            }
        )
    return records


def responses_conversation_to_chat(
    conversation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert official compact history into a valid native-tool chat history.

    The official harness uses both of these representations:

    * a compressed assistant message whose ``tool_calls`` already include
      results; and
    * an in-turn assistant message followed by a duplicate ``role=tool``
      summary.

    We emit one assistant native-call item plus one native tool-result item per
    call in either case.  A following duplicate summary is validated and
    consumed, never emitted twice.
    """

    if not isinstance(conversation, (list, tuple)):
        raise TypeError("conversation must be a list or tuple")
    converted: list[dict[str, Any]] = []
    pending_duplicate: list[dict[str, Any]] | None = None

    for message_index, raw_message in enumerate(conversation):
        if not isinstance(raw_message, Mapping):
            raise TypeError(f"conversation[{message_index}] must be a mapping")
        role = raw_message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported conversation role: {role!r}")

        if role == "tool":
            if set(raw_message) != {"role", "content"}:
                raise ValueError("official tool summary has unexpected fields")
            records = _official_call_records(
                raw_message.get("content"),
                subject=f"conversation[{message_index}].content",
            )
            if pending_duplicate is None or records != pending_duplicate:
                raise ValueError("orphaned or mismatched official tool summary")
            pending_duplicate = None
            continue

        # A compressed tool-bearing assistant is not followed by a duplicate
        # tool summary.  Reaching another normal message closes that option.
        pending_duplicate = None
        content = raw_message.get("content")
        if content is not None and not isinstance(content, str):
            raise TypeError(f"conversation[{message_index}].content must be text or null")

        if role in {"system", "user"}:
            if set(raw_message) != {"role", "content"}:
                raise ValueError(f"official {role} message has unexpected fields")
            if not isinstance(content, str):
                raise TypeError(f"official {role} content must be text")
            converted.append({"role": role, "content": content})
            continue

        allowed = {"role", "content", "tool_calls"}
        if not set(raw_message).issubset(allowed):
            raise ValueError("official assistant message has unexpected fields")
        records = _official_call_records(
            raw_message.get("tool_calls"),
            subject=f"conversation[{message_index}].tool_calls",
        )
        if not records:
            if not isinstance(content, str):
                raise TypeError("assistant text response must contain text")
            converted.append({"role": "assistant", "content": content})
            continue

        native_calls: list[dict[str, Any]] = []
        native_results: list[dict[str, Any]] = []
        for call_index, record in enumerate(records):
            call_id = f"state-call-{message_index + 1}-{call_index + 1}"
            native_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": record["name"],
                        "arguments": json.dumps(
                            record["arguments"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }
            )
            native_results.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        record["result"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        # An immediately following official tool summary marks an expanded
        # in-turn record. Preserve mixed pre-execution narration on the native
        # assistant call. Without that marker this is compact cross-turn
        # history, whose content is final narration generated after all calls.
        next_message = (
            conversation[message_index + 1]
            if message_index + 1 < len(conversation)
            else None
        )
        expanded_in_turn = (
            isinstance(next_message, Mapping)
            and next_message.get("role") == "tool"
        )
        converted.append(
            {
                "role": "assistant",
                "content": content if expanded_in_turn and content else None,
                "tool_calls": native_calls,
            }
        )
        converted.extend(native_results)
        if content and not expanded_in_turn:
            converted.append({"role": "assistant", "content": content})
        pending_duplicate = records

    return converted


@dataclass(frozen=True)
class CompletionAudit:
    """No-prompt telemetry proving the physical/replayed completion identity."""

    participant: str
    ordinal: int
    model: str
    request_sha256: str | None
    completion_sha256: str | None
    cache_hit: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _completion_audit(
    *,
    participant: str,
    ordinal: int,
    backend: Any,
    completion: Completion,
) -> CompletionAudit:
    return CompletionAudit(
        participant=participant,
        ordinal=ordinal,
        model=completion.model,
        request_sha256=getattr(backend, "last_request_digest", None),
        completion_sha256=getattr(backend, "last_completion_digest", None),
        cache_hit=bool(getattr(backend, "last_cache_hit", False)),
        latency_s=float(completion.latency_s),
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
        total_tokens=completion.usage.total_tokens,
    )


class HarnessChatAgent(BaseAgent):
    """Harness-executed STATE-Bench agent backed by any CompletionBackend."""

    def __init__(
        self,
        *,
        backend: Any,
        runtime_context: AgentRuntimeContext,
        seed: int | None = None,
    ) -> None:
        super().__init__(runtime_context=runtime_context)
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must provide complete()")
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be int or None")
        self.backend = backend
        self.seed = seed
        self._completion_audits: list[CompletionAudit] = []

    @property
    def completion_audits(self) -> tuple[CompletionAudit, ...]:
        return tuple(self._completion_audits)

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        messages = [
            {"role": "system", "content": system_prompt},
            *responses_conversation_to_chat(conversation),
        ]
        chat_tools = responses_tool_schemas_to_chat(tools)
        completion = self.backend.complete(
            messages=messages,
            tools=chat_tools or None,
            seed=self.seed,
        )
        if not isinstance(completion, Completion):
            raise TypeError("backend.complete() must return Completion")
        self.add_token_usage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
        self.total_output_tokens += completion.usage.completion_tokens
        self._completion_audits.append(
            _completion_audit(
                participant="producer",
                ordinal=len(self._completion_audits) + 1,
                backend=self.backend,
                completion=completion,
            )
        )
        response_text = completion.content or ""
        if completion.tool_calls:
            raw_text = completion.raw_content or response_text
            has_text = bool(raw_text.strip())
            if has_text != completion.provenance.mixed_content_and_tool_calls:
                raise ValueError(
                    "provider mixed-content provenance is inconsistent with its tool response"
                )
            if completion.content and completion.raw_content and (
                completion.content != completion.raw_content
            ):
                raise ValueError("provider content and raw_content disagree")
            response_text = raw_text
        return AgentTurnResponse(
            text=response_text,
            tool_calls=[
                AgentToolCallRequest(
                    name=call.name,
                    arguments=_json_clone(
                        call.arguments,
                        subject=f"completion tool {call.name!r} arguments",
                    ),
                )
                for call in completion.tool_calls
            ],
        )


class HarnessAgentFactory:
    """Agent factory that captures only a backend and generation settings."""

    def __init__(self, *, backend: Any, seed: int | None = None) -> None:
        self.backend = backend
        self.seed = seed
        self.agent: HarnessChatAgent | None = None

    def __call__(self, runtime_context: AgentRuntimeContext) -> HarnessChatAgent:
        self.agent = HarnessChatAgent(
            backend=self.backend,
            runtime_context=runtime_context,
            seed=self.seed,
        )
        return self.agent


class BackendUserSimulator:
    """Backend-neutral wrapper with byte-equivalent official prompt assembly."""

    def __init__(self, *, backend: Any, system_prompt: str, seed: int | None = None) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must provide complete()")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be int or None")
        self.backend = backend
        self.system_prompt = system_prompt
        self.seed = seed
        self._completion_audits: list[CompletionAudit] = []

    @property
    def completion_audits(self) -> tuple[CompletionAudit, ...]:
        return tuple(self._completion_audits)

    @staticmethod
    def request_messages(
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Replicate pinned ``state_bench.simulator.UserSimulator.respond``."""

        lines: list[str] = []
        for index, msg in enumerate(conversation):
            if not isinstance(msg, Mapping):
                raise TypeError(f"conversation[{index}] must be a mapping")
            role_value = msg.get("role")
            if not isinstance(role_value, str):
                raise TypeError(f"conversation[{index}].role must be text")
            role = role_value.upper()
            content = msg.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise TypeError(f"conversation[{index}].content must be text or null")
            tool_calls = msg.get("tool_calls")

            if role == "ASSISTANT" and tool_calls:
                records = _official_call_records(
                    tool_calls,
                    subject=f"conversation[{index}].tool_calls",
                    preserve_argument_order=True,
                )
                tc_summary = "\n".join(
                    f"[Called {tc['name']}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)[:200]})]"
                    for tc in records
                )
                content = f"{tc_summary}\n{content}" if content else tc_summary

            if content:
                lines.append(f"{role}: {content}")

        conversation_text = "\n\n".join(lines)
        instruction = (
            f"CONVERSATION SO FAR:\n{conversation_text}\n\n"
            "Respond as the customer based on the conversation and your rules above.\n"
            "YOUR RESPONSE:"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

    def respond(self, conversation: list[dict[str, Any]]) -> str:
        messages = self.request_messages(
            system_prompt=self.system_prompt,
            conversation=conversation,
        )
        completion = self.backend.complete(messages=messages, seed=self.seed)
        if not isinstance(completion, Completion):
            raise TypeError("backend.complete() must return Completion")
        if completion.tool_calls:
            raise ValueError("user simulator must not return tool calls")
        if not isinstance(completion.content, str) or not completion.content.strip():
            raise ValueError("user simulator returned empty text")
        self._completion_audits.append(
            _completion_audit(
                participant="user_simulator",
                ordinal=len(self._completion_audits) + 1,
                backend=self.backend,
                completion=completion,
            )
        )
        return completion.content.strip()
