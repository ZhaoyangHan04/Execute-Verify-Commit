"""Small, strict DashScope OpenAI-compatible backend.

The backend deliberately has no API-key constructor argument.  Production
credentials are read once from ``DASHSCOPE_API_KEY`` when the default client is
created and are never retained in completion provenance.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .model import Completion, CompletionProvenance, ToolCall, Usage


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DashScope advertises Kimi-K3 through the compatible endpoint, but rejects
# the otherwise neutral ``top_p=1.0`` used by the frozen workflows.  Keep the
# compatibility exception exact and model-local so Qwen user simulators and
# every previously evaluated model retain byte-for-byte request parameters.
_MODEL_TOP_P_COMPATIBILITY = {("kimi-k3", 1.0): 0.95}

_USAGE_LEDGER_ENV = "SHADOW_API_USAGE_LEDGER"
_USAGE_CONTEXT_ENV = {
    "experiment": "SHADOW_API_USAGE_EXPERIMENT",
    "stage": "SHADOW_API_USAGE_STAGE",
    "job": "SHADOW_API_USAGE_JOB",
}
_TRANSIENT_TRANSPORT_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectTimeout",
        "RateLimitError",
        "ReadTimeout",
        "TimeoutError",
    }
)
_REQUEST_SCHEDULE_LOCK = threading.Lock()
_NEXT_REQUEST_START: dict[str, float] = {}
USAGE_CALL_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("shadow_usage_context", default=None)
_INFLIGHT_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}


def _inflight_slot():
    """Optional experiment-local cap; absent means legacy behavior."""
    limit = int(os.environ.get("SHADOW_MAX_INFLIGHT", "0"))
    if limit <= 0:
        return nullcontext()
    with _REQUEST_SCHEDULE_LOCK:
        return _INFLIGHT_SEMAPHORES.setdefault(limit, threading.BoundedSemaphore(limit))


class BackendError(RuntimeError):
    """Base class for DashScope model-backend failures."""


class BackendConfigurationError(BackendError):
    """The backend cannot be constructed from the runtime configuration."""


class BackendTransportError(BackendError):
    """The compatible endpoint could not complete the physical request."""

    def __init__(self, message: str, *, provider_exception_type: str) -> None:
        super().__init__(message)
        self.provider_exception_type = provider_exception_type


class BackendProtocolError(BackendError):
    """The endpoint returned a response outside the frozen protocol."""

    def __init__(self, message: str, *, tool_names: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.tool_names = tool_names


@dataclass(frozen=True)
class DashScopeConfig:
    """Generation settings shared by one backend instance."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 1024
    enable_thinking: bool = False
    timeout_seconds: float = 120.0
    transport_max_attempts: int = 1
    transport_min_interval_seconds: float = 0.0
    transport_retry_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if type(self.transport_max_attempts) is not int or not (
            1 <= self.transport_max_attempts <= 10
        ):
            raise ValueError("transport_max_attempts must be an integer in [1, 10]")
        if self.transport_min_interval_seconds < 0:
            raise ValueError("transport_min_interval_seconds must be non-negative")
        if self.transport_retry_backoff_seconds < 0:
            raise ValueError("transport_retry_backoff_seconds must be non-negative")

    def replay_request_identity(self) -> dict[str, Any]:
        """Return only provider-request fields used by response replay.

        Retry and pacing settings govern how an unchanged request reaches the
        provider.  They do not change the request or its successful response,
        so including them in the replay key would invalidate compatible caches.
        Keep this shape identical to DashScopeConfig before transport controls
        were added.
        """

        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_completion_tokens": self.max_completion_tokens,
            "enable_thinking": self.enable_thinking,
            "timeout_seconds": self.timeout_seconds,
        }


def _strict_json_object(raw: str, *, subject: str) -> dict[str, Any]:
    """Decode standards-compliant JSON without duplicate object keys."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, ValueError):
        raise BackendProtocolError(f"{subject} has invalid strict JSON") from None
    if not _all_numbers_finite(value):
        raise BackendProtocolError(f"{subject} has non-finite JSON numbers")
    if not isinstance(value, dict):
        raise BackendProtocolError(f"{subject} must be a JSON object")
    return value


def _all_numbers_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    return True


class DashScopeBackend:
    """Strict one-model backend with no fallback or hidden SDK retries."""

    def __init__(
        self,
        model: str,
        config: DashScopeConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self.model = model.strip()
        requested_config = config or DashScopeConfig()
        compatible_top_p = _MODEL_TOP_P_COMPATIBILITY.get(
            (self.model, requested_config.top_p)
        )
        self.config = (
            replace(requested_config, top_p=compatible_top_p)
            if compatible_top_p is not None
            else requested_config
        )
        self._client = client if client is not None else self._create_client()

    def _create_client(self) -> Any:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise BackendConfigurationError(
                "DASHSCOPE_API_KEY must be present in the process environment"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise BackendConfigurationError(
                "the openai package is required for the DashScope backend"
            ) from None
        return OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
            max_retries=0,
            timeout=self.config.timeout_seconds,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Perform one physical completion and validate its native shape."""
        self._validate_request(messages, tools, seed, json_mode)
        self._tool_names(tools)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_completion_tokens": self.config.max_completion_tokens,
            "extra_body": {"enable_thinking": self.config.enable_thinking},
        }
        if seed is not None:
            request["seed"] = seed
        # Kimi's paper configuration uses native sampling, not an imposed T=0.
        # Outer task seeds remain unchanged; the model request has no seed.
        if self.model == "kimi-k3":
            for parameter in ("temperature", "top_p", "seed"):
                request.pop(parameter, None)
        if tools:
            request["tools"] = [dict(tool) for tool in tools]
            request["tool_choice"] = "auto"
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        request_mode: Literal["text", "json", "tools"] = (
            "tools" if tools else "json" if json_mode else "text"
        )
        logical_request_id = f"{os.getpid()}-{time.time_ns()}"
        for attempt_index in range(1, self.config.transport_max_attempts + 1):
            self._wait_for_request_slot()
            started_at_epoch_s = time.time()
            started = time.perf_counter()
            try:
                with _inflight_slot():
                    response = self._client.chat.completions.create(**request)
            except Exception as exc:
                latency_s = time.perf_counter() - started
                error_type = type(exc).__name__
                retry_scheduled = (
                    error_type in getattr(self, "transient_transport_errors", _TRANSIENT_TRANSPORT_ERRORS)
                    and attempt_index < self.config.transport_max_attempts
                )
                retry_delay_s = (
                    self.config.transport_retry_backoff_seconds * attempt_index
                    if retry_scheduled
                    else 0.0
                )
                self._record_physical_request(
                    status="transport_error",
                    started_at_epoch_s=started_at_epoch_s,
                    latency_s=latency_s,
                    request_mode=request_mode,
                    seed=seed,
                    error_type=error_type,
                    logical_request_id=logical_request_id,
                    attempt_index=attempt_index,
                    retry_scheduled=retry_scheduled,
                    retry_delay_s=retry_delay_s,
                )
                if retry_scheduled:
                    if retry_delay_s:
                        time.sleep(retry_delay_s)
                    continue
                # Do not propagate provider exception text: it may contain request
                # headers or other sensitive transport details.
                raise BackendTransportError(
                    f"DashScope request failed ({error_type})",
                    provider_exception_type=error_type,
                ) from None
            latency_s = time.perf_counter() - started
            try:
                completion = self._parse_response(
                    response,
                    latency_s=latency_s,
                    tools=tools,
                    seed=seed,
                    json_mode=json_mode,
                )
            except Exception as exc:
                self._record_physical_request(
                    status=(
                        "protocol_error"
                        if isinstance(exc, BackendProtocolError)
                        else "response_parse_error"
                    ),
                    started_at_epoch_s=started_at_epoch_s,
                    latency_s=latency_s,
                    request_mode=request_mode,
                    seed=seed,
                    response=response,
                    error_type=type(exc).__name__,
                    logical_request_id=logical_request_id,
                    attempt_index=attempt_index,
                )
                raise
            self._record_physical_request(
                status="success",
                started_at_epoch_s=started_at_epoch_s,
                latency_s=latency_s,
                request_mode=request_mode,
                seed=seed,
                response=response,
                completion=completion,
                logical_request_id=logical_request_id,
                attempt_index=attempt_index,
            )
            return completion
        raise AssertionError("transport attempt loop ended without a result")

    def _wait_for_request_slot(self) -> None:
        interval = self.config.transport_min_interval_seconds
        if interval <= 0:
            return
        with _REQUEST_SCHEDULE_LOCK:
            now = time.monotonic()
            scheduled = max(now, _NEXT_REQUEST_START.get(self.model, now))
            _NEXT_REQUEST_START[self.model] = scheduled + interval
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)

    def _record_physical_request(
        self,
        *,
        status: str,
        started_at_epoch_s: float,
        latency_s: float,
        request_mode: Literal["text", "json", "tools"],
        seed: int | None,
        response: Any | None = None,
        completion: Completion | None = None,
        error_type: str | None = None,
        logical_request_id: str | None = None,
        attempt_index: int = 1,
        retry_scheduled: bool = False,
        retry_delay_s: float = 0.0,
    ) -> None:
        """Append one redacted cost record when experiment accounting is enabled.

        The ledger is deliberately best-effort and contains no messages, tool
        arguments, credentials, provider exception text, or completion text.
        One compact ``O_APPEND`` write lets independent worker processes share
        the same JSONL ledger without changing benchmark behavior.
        """

        ledger = os.environ.get(_USAGE_LEDGER_ENV, "").strip()
        if not ledger:
            return
        if completion is not None:
            returned_usage = self._best_effort_usage(response)
            usage_present = all(value is not None for value in returned_usage)
            if usage_present:
                prompt_tokens, completion_tokens, total_tokens = returned_usage
                if (
                    prompt_tokens != completion.usage.prompt_tokens
                    or completion_tokens != completion.usage.completion_tokens
                    or total_tokens != completion.usage.total_tokens
                ):
                    usage_present = False
                    prompt_tokens = completion_tokens = total_tokens = None
            else:
                prompt_tokens = completion_tokens = total_tokens = None
            response_model: str | None = completion.model
        else:
            returned_usage = self._best_effort_usage(response)
            usage_present = all(value is not None for value in returned_usage)
            if usage_present:
                prompt_tokens, completion_tokens, total_tokens = returned_usage
            else:
                prompt_tokens = completion_tokens = total_tokens = None
            raw_response_model = getattr(response, "model", None)
            response_model = (
                raw_response_model if isinstance(raw_response_model, str) else None
            )
        ended_at_epoch_s = started_at_epoch_s + latency_s
        record: dict[str, Any] = {
            "schema": "shadow-verifier.api-usage.v1",
            "event_id": f"{os.getpid()}-{time.time_ns()}",
            "logical_request_id": logical_request_id,
            "attempt_index": attempt_index,
            "max_attempts": self.config.transport_max_attempts,
            "retry_scheduled": retry_scheduled,
            "retry_delay_s": retry_delay_s,
            "configured_min_interval_seconds": (
                self.config.transport_min_interval_seconds
            ),
            "pid": os.getpid(),
            "status": status,
            "started_at_epoch_s": started_at_epoch_s,
            "ended_at_epoch_s": ended_at_epoch_s,
            "latency_s": latency_s,
            "requested_model": self.model,
            "response_model": response_model,
            "request_mode": request_mode,
            "seed": seed,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_completion_tokens": self.config.max_completion_tokens,
            "enable_thinking": self.config.enable_thinking,
            "usage_present": usage_present,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "error_type": error_type,
            "finish_reason": self._best_effort_finish_reason(response),
            "response_content_chars": self._best_effort_message_chars(
                response, "content"
            ),
            "response_reasoning_chars": self._best_effort_message_chars(
                response, "reasoning_content"
            ),
            "reasoning_tokens": self._best_effort_reasoning_tokens(response),
        }
        for field, environment_name in _USAGE_CONTEXT_ENV.items():
            value = os.environ.get(environment_name, "").strip()
            record[field] = value or None
        metadata = getattr(self, "request_metadata", None)
        if isinstance(metadata, dict):
            record["transport_metadata"] = dict(metadata)
        call_context = USAGE_CALL_CONTEXT.get()
        if call_context is not None:
            record["call_context"] = dict(call_context)
        payload = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        try:
            path = Path(ledger)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)
        except OSError:
            # Accounting must never alter a model decision or turn a successful
            # benchmark request into a workflow failure.
            return

    @staticmethod
    def _best_effort_usage(response: Any | None) -> tuple[int | None, int | None, int | None]:
        usage = getattr(response, "usage", None)
        values: list[int | None] = []
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, name, None) if usage is not None else None
            values.append(value if type(value) is int and value >= 0 else None)
        return values[0], values[1], values[2]

    @staticmethod
    def _best_effort_finish_reason(response: Any | None) -> str | None:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            return None
        value = getattr(choices[0], "finish_reason", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _best_effort_message_chars(
        response: Any | None,
        field: str,
    ) -> int | None:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            return None
        message = getattr(choices[0], "message", None)
        value = getattr(message, field, None) if message is not None else None
        return len(value) if isinstance(value, str) else None

    @staticmethod
    def _best_effort_reasoning_tokens(response: Any | None) -> int | None:
        usage = getattr(response, "usage", None)
        details = (
            getattr(usage, "completion_tokens_details", None)
            if usage is not None
            else None
        )
        value = getattr(details, "reasoning_tokens", None)
        return value if type(value) is int and value >= 0 else None

    @staticmethod
    def _validate_request(
        messages: object,
        tools: object,
        seed: object,
        json_mode: object,
    ) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if not all(isinstance(message, dict) for message in messages):
            raise TypeError("messages must contain only dictionaries")
        if tools is not None and (
            not isinstance(tools, list)
            or not all(isinstance(tool, dict) for tool in tools)
        ):
            raise TypeError("tools must be a list of dictionaries or None")
        if type(json_mode) is not bool:
            raise TypeError("json_mode must be exactly bool")
        if json_mode and tools:
            raise ValueError("json_mode and native tools are mutually exclusive")
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be int or None")

    def _parse_response(
        self,
        response: Any,
        *,
        latency_s: float,
        tools: list[dict[str, Any]] | None,
        seed: int | None,
        json_mode: bool,
    ) -> Completion:
        response_model = getattr(response, "model", None)
        if response_model != self.model:
            raise BackendProtocolError(
                "model identity mismatch: endpoint did not return the requested model"
            )
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise BackendProtocolError("response must contain exactly one choice")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise BackendProtocolError("response choice is missing its message")

        raw_content = getattr(message, "content", None)
        if raw_content is not None and not isinstance(raw_content, str):
            raise BackendProtocolError("response content must be text or null")
        raw_calls = getattr(message, "tool_calls", None)
        if raw_calls is None:
            raw_calls = ()
        elif not isinstance(raw_calls, (list, tuple)):
            raise BackendProtocolError("response tool_calls must be a sequence")
        parsed_calls = tuple(self._parse_tool_call(call) for call in raw_calls)
        if len({call.id for call in parsed_calls}) != len(parsed_calls):
            raise BackendProtocolError("tool call ids must be unique")
        allowed_tool_names = self._tool_names(tools)
        unknown_names = {
            call.name for call in parsed_calls if call.name not in allowed_tool_names
        }
        if unknown_names:
            raise BackendProtocolError(
                "response called a function outside the supplied tool schemas",
                tool_names=tuple(sorted(unknown_names)),
            )

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "tool_calls":
            if not parsed_calls:
                raise BackendProtocolError(
                    "finish_reason='tool_calls' requires at least one tool call"
                )
        elif finish_reason == "stop":
            if parsed_calls:
                raise BackendProtocolError(
                    "native tool calls require finish_reason='tool_calls'"
                )
            if raw_content is None or not raw_content.strip():
                raise BackendProtocolError("text completion is empty")
        else:
            raise BackendProtocolError(
                f"unsupported finish_reason: {finish_reason!r}"
            )

        usage_raw = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=self._usage_value(usage_raw, "prompt_tokens"),
            completion_tokens=self._usage_value(
                usage_raw, "completion_tokens"
            ),
            total_tokens=self._usage_value(usage_raw, "total_tokens"),
        )
        request_mode: Literal["text", "json", "tools"] = (
            "tools" if tools else "json" if json_mode else "text"
        )
        provenance = CompletionProvenance(
            requested_model=self.model,
            response_model=response_model,
            request_mode=request_mode,
            mixed_content_and_tool_calls=bool(
                parsed_calls and raw_content and raw_content.strip()
            ),
            tool_schema_count=len(tools or ()),
            tool_call_count=len(parsed_calls),
            seed=seed,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_completion_tokens=self.config.max_completion_tokens,
            enable_thinking=self.config.enable_thinking,
        )
        return Completion(
            # When native calls are present, only those calls are executable
            # assistant content.  Keep any provider text solely as provenance
            # so a harness cannot publish mixed narration by accident.
            content=None if parsed_calls else raw_content,
            tool_calls=parsed_calls,
            finish_reason=finish_reason,
            model=response_model,
            usage=usage,
            latency_s=latency_s,
            raw_content=raw_content,
            provenance=provenance,
        )

    @staticmethod
    def _parse_tool_call(raw_call: Any) -> ToolCall:
        call_type = getattr(raw_call, "type", None)
        call_id = getattr(raw_call, "id", None)
        function = getattr(raw_call, "function", None)
        name = getattr(function, "name", None) if function is not None else None
        raw_arguments = (
            getattr(function, "arguments", None) if function is not None else None
        )
        if call_type != "function":
            raise BackendProtocolError("only native function tool calls are supported")
        if not isinstance(call_id, str) or not call_id:
            raise BackendProtocolError("tool call id must be non-empty")
        if not isinstance(name, str) or not name:
            raise BackendProtocolError("tool call name must be non-empty")
        if not isinstance(raw_arguments, str):
            raise BackendProtocolError("tool call arguments must be JSON text")
        arguments = _strict_json_object(
            raw_arguments,
            subject=f"tool call {name!r} arguments",
        )
        return ToolCall(id=call_id, name=name, arguments=arguments)

    @staticmethod
    def _tool_names(tools: list[dict[str, Any]] | None) -> frozenset[str]:
        names: set[str] = set()
        for tool in tools or ():
            if tool.get("type") != "function":
                raise BackendProtocolError(
                    "only native function tool schemas are supported"
                )
            function = tool.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name:
                raise BackendProtocolError("tool schema name must be non-empty")
            if name in names:
                raise BackendProtocolError("tool schema names must be unique")
            names.add(name)
        return frozenset(names)

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int:
        value = getattr(usage, name, 0) if usage is not None else 0
        if value is None:
            return 0
        if type(value) is not int or value < 0:
            raise BackendProtocolError(f"usage.{name} must be a non-negative int")
        return value
