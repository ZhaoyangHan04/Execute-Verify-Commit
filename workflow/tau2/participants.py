"""Official τ² participant state machines backed by an injected chat backend."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterable

from shadow_verifier.backends.dashscope import DashScopeBackend


def _openai_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    from tau2.data_model.message import (
        AssistantMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )

    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            row: dict[str, Any] = {"role": "user", "content": message.content}
            if message.tool_calls:
                row["tool_calls"] = _tool_calls_payload(message.tool_calls)
            converted.append(row)
        elif isinstance(message, AssistantMessage):
            row = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                row["tool_calls"] = _tool_calls_payload(message.tool_calls)
            converted.append(row)
        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        else:
            raise TypeError(f"unsupported τ² message type: {type(message)!r}")
    return converted


def _tool_calls_payload(calls: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    # Replay cache entries are canonical JSON, so decoding a
                    # cached completion can reorder object keys.  Canonicalize
                    # live calls too: otherwise semantically identical tool
                    # arguments produce different next-turn request bytes in
                    # the baseline and replay arms.
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
        for call in calls
    ]


def build_native_agent(
    *,
    tools: list[Any],
    domain_policy: str,
    backend: DashScopeBackend,
):
    """Build the official LLMAgent state machine with an injected provider."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        UserMessage,
    )

    class NativeAgent(LLMAgent):
        def __init__(self) -> None:
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                llm=backend.model,
                llm_args={},
            )
            self.backend = backend

        def _generate_next_message(self, message, state):
            if isinstance(message, UserMessage) and message.is_audio:
                raise ValueError("audio is outside the text workflow scope")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            completion = self.backend.complete(
                messages=_openai_messages(state.system_messages + state.messages),
                tools=[tool.openai_schema for tool in self.tools],
                seed=self.llm_args.get("seed"),
            )
            calls = tuple(
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    requestor="assistant",
                )
                for call in completion.tool_calls
            )
            # τ² half-duplex requires text XOR tool calls. Some providers emit
            # both; preserve raw content in backend provenance but expose tools.
            content = None if calls else completion.content
            provenance = asdict(completion.provenance)
            provenance["replay_cache_hit"] = bool(
                getattr(self.backend, "last_cache_hit", False)
            )
            provenance["replay_request_sha256"] = getattr(
                self.backend, "last_request_digest", None
            )
            provenance["replay_completion_sha256"] = getattr(
                self.backend, "last_completion_digest", None
            )
            return AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=list(calls) or None,
                usage=asdict(completion.usage),
                raw_data=provenance,
                generation_time_seconds=completion.latency_s,
            )

    return NativeAgent()


def build_native_user(
    *,
    instructions: str,
    tools: list[Any] | None,
    backend: DashScopeBackend,
):
    """Build the official UserSimulator state machine with an injected provider."""
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.user.user_simulator import UserSimulator

    class NativeUser(UserSimulator):
        def __init__(self) -> None:
            super().__init__(
                llm=backend.model,
                instructions=instructions,
                tools=tools,
                llm_args={},
            )
            self.backend = backend

        def _generate_next_message(self, message, state):
            if isinstance(message, AssistantMessage) and message.is_audio:
                raise ValueError("audio is outside the text workflow scope")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif isinstance(message, ToolMessage):
                state.messages.append(message)
            elif message.has_content() or message.is_tool_call():
                state.messages.append(message)
            completion = self.backend.complete(
                messages=_openai_messages(state.system_messages + state.flip_roles()),
                tools=[tool.openai_schema for tool in self.tools or []],
                seed=self.llm_args.get("seed"),
            )
            calls = tuple(
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    requestor="user",
                )
                for call in completion.tool_calls
            )
            content = None if calls else completion.content
            provenance = asdict(completion.provenance)
            provenance["replay_cache_hit"] = bool(
                getattr(self.backend, "last_cache_hit", False)
            )
            provenance["replay_request_sha256"] = getattr(
                self.backend, "last_request_digest", None
            )
            provenance["replay_completion_sha256"] = getattr(
                self.backend, "last_completion_digest", None
            )
            return UserMessage(
                role="user",
                content=content,
                tool_calls=list(calls) or None,
                usage=asdict(completion.usage),
                raw_data=provenance,
                generation_time_seconds=completion.latency_s,
            )

    return NativeUser()
