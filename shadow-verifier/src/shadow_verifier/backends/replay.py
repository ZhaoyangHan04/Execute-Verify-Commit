"""Strict, local response replay for paired experiment arms.

Only successful :class:`Completion` values are persisted.  Request material is
used solely to derive the filename, so prompts and credentials are never
written to the cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import threading
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .model import Completion, CompletionProvenance, ToolCall, Usage


_MAX_ENTRY_BYTES = 32 * 1024 * 1024


class ReplayCacheError(RuntimeError):
    """Base error raised by the response replay layer."""


class ReplayCacheCorruptionError(ReplayCacheError):
    """A present cache entry cannot be trusted or reconstructed."""


class ReplayCacheMissError(ReplayCacheError):
    """Replay-only mode could not find the exact requested completion."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _numbers_are_finite(value: object) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_numbers_are_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_numbers_are_finite(item) for item in value.values())
    return True


def _json_value(value: Any, *, subject: str) -> Any:
    """Copy a value into the exact, finite JSON subset used for keying."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{subject} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, subject=f"{subject}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{subject} contains a non-string object key")
            result[key] = _json_value(item, subject=f"{subject}.{key}")
        return result
    raise TypeError(f"{subject} contains a non-JSON value")


def _exact_fields(value: object, expected: set[str], *, subject: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ReplayCacheCorruptionError(f"{subject} must be an object")
    if set(value) != expected:
        raise ReplayCacheCorruptionError(f"{subject} has missing or extra fields")
    return value


def _strict_str(value: object, *, subject: str, nonempty: bool = False) -> str:
    if type(value) is not str or (nonempty and not value):
        qualifier = "a non-empty string" if nonempty else "a string"
        raise ReplayCacheCorruptionError(f"{subject} must be {qualifier}")
    return value


def _optional_str(value: object, *, subject: str) -> str | None:
    if value is not None and type(value) is not str:
        raise ReplayCacheCorruptionError(f"{subject} must be a string or null")
    return value


def _strict_int(value: object, *, subject: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ReplayCacheCorruptionError(f"{subject} must be a valid integer")
    return value


def _strict_float(value: object, *, subject: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ReplayCacheCorruptionError(f"{subject} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ReplayCacheCorruptionError(f"{subject} must be finite and in range")
    return result


class ReplayCacheBackend:
    """Wrap a completion backend with strict filesystem response replay."""

    def __init__(
        self,
        backend: Any,
        cache_dir: Path,
        replay_only: bool = False,
    ) -> None:
        model = getattr(backend, "model", None)
        if type(model) is not str or not model:
            raise TypeError("backend.model must be a non-empty string")
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend.complete must be callable")
        if not hasattr(backend, "config"):
            raise TypeError("backend.config is required for strict replay identity")
        if not isinstance(cache_dir, Path):
            raise TypeError("cache_dir must be a pathlib.Path")
        if type(replay_only) is not bool:
            raise TypeError("replay_only must be exactly bool")

        self._backend = backend
        self.model = model
        self.config = backend.config
        self.cache_dir = cache_dir
        self.last_cache_hit = False
        self.last_request_digest: str | None = None
        self.last_completion_digest: str | None = None
        self._replay_only = replay_only
        self._lock = threading.RLock()

        try:
            cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = cache_dir.lstat()
        except OSError as exc:
            raise ReplayCacheError("cannot initialize replay cache directory") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ReplayCacheError("replay cache path must be a real directory")

    @property
    def replay_only(self) -> bool:
        """Whether a cache miss is forbidden from reaching the backend."""

        with self._lock:
            return self._replay_only

    def release_replay_only(self) -> None:
        """Irreversibly allow this instance to populate missing entries."""

        with self._lock:
            self._replay_only = False

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Replay an exact request, otherwise call and cache the wrapped backend."""

        key = self._request_key(messages, tools, seed, json_mode)
        path = self.cache_dir / f"{key}.json"
        with self._lock:
            self.last_cache_hit = False
            self.last_request_digest = key
            self.last_completion_digest = None
            if path.exists() or path.is_symlink():
                completion, completion_digest = self._read(path)
                self._validate_completion(
                    completion,
                    tools=tools,
                    seed=seed,
                    json_mode=json_mode,
                )
                self.last_completion_digest = completion_digest
                self.last_cache_hit = True
                return replace(completion, latency_s=0.0)

            if self._replay_only:
                raise ReplayCacheMissError(
                    "exact completion is absent while replay-only mode is active"
                )
            completion = self._backend.complete(
                messages,
                tools=tools,
                seed=seed,
                json_mode=json_mode,
            )
            self._validate_completion(
                completion,
                tools=tools,
                seed=seed,
                json_mode=json_mode,
            )
            self._write(path, completion)
            self.last_completion_digest = self._completion_digest(completion)
            return completion

    def _request_key(
        self,
        messages: object,
        tools: object,
        seed: object,
        json_mode: object,
    ) -> str:
        config = self.config
        identity_factory = getattr(config, "replay_request_identity", None)
        if callable(identity_factory):
            config_value = identity_factory()
            if type(config_value) is not dict:
                raise TypeError("config replay_request_identity() must return a dictionary")
        elif is_dataclass(config) and not isinstance(config, type):
            config_value = asdict(config)
        elif type(config) is dict:
            config_value = config
        else:
            raise TypeError("backend.config must be a dataclass instance or dictionary")
        payload = {
            "model": self.model,
            "config": {
                "type": f"{type(config).__module__}.{type(config).__qualname__}",
                "value": _json_value(config_value, subject="config"),
            },
            "messages": _json_value(messages, subject="messages"),
            "tools": _json_value(tools, subject="tools"),
            "seed": _json_value(seed, subject="seed"),
            "json_mode": _json_value(json_mode, subject="json_mode"),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read(self, path: Path) -> tuple[Completion, str]:
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ReplayCacheCorruptionError("cache entry is not a regular file")
            if info.st_mode & 0o077:
                raise ReplayCacheCorruptionError("cache entry permissions exceed 0600")
            if info.st_size > _MAX_ENTRY_BYTES:
                raise ReplayCacheCorruptionError("cache entry is unreasonably large")
            raw = path.read_text(encoding="utf-8")
            value = json.loads(
                raw,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except ReplayCacheCorruptionError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ReplayCacheCorruptionError("cache entry is not strict JSON") from exc
        if not _numbers_are_finite(value):
            raise ReplayCacheCorruptionError("cache entry contains non-finite numbers")
        completion = self._decode_completion(value)
        digest = hashlib.sha256(self._canonical_bytes(value)).hexdigest()
        return completion, digest

    @staticmethod
    def _decode_completion(value: object) -> Completion:
        top = _exact_fields(
            value,
            {field.name for field in fields(Completion)},
            subject="completion",
        )
        raw_calls = top["tool_calls"]
        if type(raw_calls) is not list:
            raise ReplayCacheCorruptionError("completion.tool_calls must be a list")
        calls: list[ToolCall] = []
        for index, raw_call in enumerate(raw_calls):
            call = _exact_fields(
                raw_call,
                {field.name for field in fields(ToolCall)},
                subject=f"completion.tool_calls[{index}]",
            )
            arguments = call["arguments"]
            if type(arguments) is not dict:
                raise ReplayCacheCorruptionError("tool-call arguments must be an object")
            calls.append(
                ToolCall(
                    id=_strict_str(call["id"], subject="tool-call id", nonempty=True),
                    name=_strict_str(call["name"], subject="tool-call name", nonempty=True),
                    arguments=arguments,
                )
            )

        raw_usage = _exact_fields(
            top["usage"],
            {field.name for field in fields(Usage)},
            subject="completion.usage",
        )
        usage = Usage(
            prompt_tokens=_strict_int(
                raw_usage["prompt_tokens"],
                subject="usage.prompt_tokens",
                minimum=0,
            ),
            completion_tokens=_strict_int(
                raw_usage["completion_tokens"],
                subject="usage.completion_tokens",
                minimum=0,
            ),
            total_tokens=_strict_int(
                raw_usage["total_tokens"],
                subject="usage.total_tokens",
                minimum=0,
            ),
        )

        raw_provenance = _exact_fields(
            top["provenance"],
            {field.name for field in fields(CompletionProvenance)},
            subject="completion.provenance",
        )
        mixed = raw_provenance["mixed_content_and_tool_calls"]
        thinking = raw_provenance["enable_thinking"]
        if type(mixed) is not bool or type(thinking) is not bool:
            raise ReplayCacheCorruptionError("provenance booleans must be exactly bool")
        seed = raw_provenance["seed"]
        if seed is not None:
            seed = _strict_int(seed, subject="provenance.seed")
        provenance = CompletionProvenance(
            requested_model=_strict_str(
                raw_provenance["requested_model"],
                subject="provenance.requested_model",
                nonempty=True,
            ),
            response_model=_strict_str(
                raw_provenance["response_model"],
                subject="provenance.response_model",
                nonempty=True,
            ),
            request_mode=_strict_str(
                raw_provenance["request_mode"],
                subject="provenance.request_mode",
            ),
            mixed_content_and_tool_calls=mixed,
            tool_schema_count=_strict_int(
                raw_provenance["tool_schema_count"],
                subject="provenance.tool_schema_count",
                minimum=0,
            ),
            tool_call_count=_strict_int(
                raw_provenance["tool_call_count"],
                subject="provenance.tool_call_count",
                minimum=0,
            ),
            seed=seed,
            temperature=_strict_float(
                raw_provenance["temperature"], subject="provenance.temperature"
            ),
            top_p=_strict_float(raw_provenance["top_p"], subject="provenance.top_p"),
            max_completion_tokens=_strict_int(
                raw_provenance["max_completion_tokens"],
                subject="provenance.max_completion_tokens",
                minimum=1,
            ),
            enable_thinking=thinking,
        )
        return Completion(
            content=_optional_str(top["content"], subject="completion.content"),
            tool_calls=tuple(calls),
            finish_reason=_strict_str(
                top["finish_reason"],
                subject="completion.finish_reason",
                nonempty=True,
            ),
            model=_strict_str(top["model"], subject="completion.model", nonempty=True),
            usage=usage,
            latency_s=_strict_float(top["latency_s"], subject="completion.latency_s", minimum=0.0),
            raw_content=_optional_str(top["raw_content"], subject="completion.raw_content"),
            provenance=provenance,
        )

    def _validate_completion(
        self,
        completion: object,
        *,
        tools: object,
        seed: object,
        json_mode: object,
    ) -> None:
        if type(completion) is not Completion:
            raise ReplayCacheCorruptionError("backend result must be exactly Completion")
        if (
            type(completion.usage) is not Usage
            or type(completion.provenance) is not CompletionProvenance
        ):
            raise ReplayCacheCorruptionError("completion nested records have invalid types")
        if type(completion.model) is not str or not completion.model:
            raise ReplayCacheCorruptionError("completion model has invalid type")
        if completion.content is not None and type(completion.content) is not str:
            raise ReplayCacheCorruptionError("completion content has invalid type")
        if completion.raw_content is not None and type(completion.raw_content) is not str:
            raise ReplayCacheCorruptionError("completion raw content has invalid type")
        if type(completion.finish_reason) is not str:
            raise ReplayCacheCorruptionError("completion finish reason has invalid type")
        if type(completion.tool_calls) is not tuple:
            raise ReplayCacheCorruptionError("completion.tool_calls must be exactly tuple")
        if any(type(call) is not ToolCall for call in completion.tool_calls):
            raise ReplayCacheCorruptionError("completion tool calls have invalid types")
        if completion.model != self.model:
            raise ReplayCacheCorruptionError("completion model identity mismatch")
        provenance = completion.provenance
        if provenance.requested_model != self.model or provenance.response_model != self.model:
            raise ReplayCacheCorruptionError("completion provenance model identity mismatch")
        if (
            type(provenance.requested_model) is not str
            or type(provenance.response_model) is not str
        ):
            raise ReplayCacheCorruptionError("completion provenance model types are invalid")
        if type(provenance.request_mode) is not str:
            raise ReplayCacheCorruptionError("completion request mode has invalid type")
        if provenance.seed is not None and type(provenance.seed) is not int:
            raise ReplayCacheCorruptionError("completion seed has invalid type")
        for name in ("tool_schema_count", "tool_call_count"):
            _strict_int(
                getattr(provenance, name),
                subject=f"provenance.{name}",
                minimum=0,
            )
        _strict_int(
            provenance.max_completion_tokens,
            subject="provenance.max_completion_tokens",
            minimum=1,
        )
        for name in ("temperature", "top_p"):
            _strict_float(getattr(provenance, name), subject=f"provenance.{name}")
        if type(provenance.enable_thinking) is not bool:
            raise ReplayCacheCorruptionError("completion thinking provenance has invalid type")
        if type(provenance.mixed_content_and_tool_calls) is not bool:
            raise ReplayCacheCorruptionError(
                "completion mixed-content provenance has invalid type"
            )

        expected_mode = "tools" if tools else "json" if json_mode else "text"
        if provenance.request_mode != expected_mode:
            raise ReplayCacheCorruptionError("completion request mode mismatch")
        if provenance.seed != seed:
            raise ReplayCacheCorruptionError("completion seed mismatch")
        expected_tool_count = len(tools or ())
        if provenance.tool_schema_count != expected_tool_count:
            raise ReplayCacheCorruptionError("completion tool schema count mismatch")
        if provenance.tool_call_count != len(completion.tool_calls):
            raise ReplayCacheCorruptionError("completion tool call count mismatch")

        for name in ("temperature", "top_p", "max_completion_tokens", "enable_thinking"):
            if hasattr(self.config, name) and getattr(provenance, name) != getattr(
                self.config, name
            ):
                raise ReplayCacheCorruptionError(f"completion config provenance mismatch: {name}")
        if len({call.id for call in completion.tool_calls}) != len(completion.tool_calls):
            raise ReplayCacheCorruptionError("completion tool-call ids must be unique")
        for call in completion.tool_calls:
            if (
                type(call.id) is not str
                or not call.id
                or type(call.name) is not str
                or not call.name
            ):
                raise ReplayCacheCorruptionError("completion tool-call identity is invalid")
            _json_value(call.arguments, subject="tool-call arguments")
        if completion.tool_calls:
            if completion.finish_reason != "tool_calls" or completion.content is not None:
                raise ReplayCacheCorruptionError("tool completion has invalid executable shape")
        else:
            if (
                completion.finish_reason != "stop"
                or type(completion.content) is not str
                or not completion.content.strip()
            ):
                raise ReplayCacheCorruptionError("text completion has invalid executable shape")
        expected_mixed = bool(
            completion.tool_calls
            and completion.raw_content
            and completion.raw_content.strip()
        )
        if provenance.mixed_content_and_tool_calls is not expected_mixed:
            raise ReplayCacheCorruptionError("completion mixed-content provenance mismatch")
        _strict_float(completion.latency_s, subject="completion.latency_s", minimum=0.0)
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            _strict_int(
                getattr(completion.usage, field_name),
                subject=f"usage.{field_name}",
                minimum=0,
            )

    @staticmethod
    def _canonical_bytes(payload: object) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def _completion_bytes(cls, completion: Completion) -> bytes:
        return cls._canonical_bytes(asdict(completion))

    @classmethod
    def _completion_digest(cls, completion: Completion) -> str:
        return hashlib.sha256(cls._completion_bytes(completion)).hexdigest()

    @classmethod
    def _write(cls, path: Path, completion: Completion) -> None:
        encoded = cls._completion_bytes(completion)
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.stem}.",
                suffix=".tmp",
                dir=path.parent,
            )
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600, follow_symlinks=False)
            temporary = None
        except OSError as exc:
            raise ReplayCacheError("cannot atomically persist replay entry") from exc
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
