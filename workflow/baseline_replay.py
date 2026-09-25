"""Audited import of baseline-only completion entries for exact replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_MAX_PAIR_BYTES = 128 * 1024 * 1024
_MAX_CACHE_BYTES = 32 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


class BaselineReplayImportError(RuntimeError):
    """A requested prior baseline could not be verified and imported."""


@dataclass(frozen=True)
class BaselineReplayReceipt:
    source_pair_path: str
    source_pair_sha256: str
    imported_entries: int


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _load_json_file(path: Path, *, max_bytes: int, subject: str) -> tuple[Any, bytes]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise BaselineReplayImportError(f"{subject} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise BaselineReplayImportError(f"{subject} must be a regular non-symlink file")
    if info.st_size > max_bytes:
        raise BaselineReplayImportError(f"{subject} is unreasonably large")
    try:
        encoded = path.read_bytes()
        value = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BaselineReplayImportError(f"{subject} is not strict JSON") from exc
    if not _finite(value):
        raise BaselineReplayImportError(f"{subject} contains non-finite values")
    return value, encoded


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _source_task_dir(roots: Iterable[Path], task_id: str) -> Path:
    matches: list[Path] = []
    for root in roots:
        if not isinstance(root, Path):
            raise TypeError("baseline source roots must be pathlib.Path values")
        resolved = root.resolve()
        task_dir = (resolved / "tasks" / task_id).resolve()
        try:
            task_dir.relative_to(resolved)
        except ValueError as exc:
            raise BaselineReplayImportError("task id escaped baseline source root") from exc
        if (task_dir / "pair.json").is_file():
            matches.append(task_dir)
    if len(matches) != 1:
        raise BaselineReplayImportError(
            f"expected exactly one baseline source for {task_id!r}, found {len(matches)}"
        )
    return matches[0]


def _atomic_copy(source: Path, destination: Path) -> None:
    value, encoded = _load_json_file(
        source,
        max_bytes=_MAX_CACHE_BYTES,
        subject="source completion cache entry",
    )
    if destination.exists() or destination.is_symlink():
        existing, _ = _load_json_file(
            destination,
            max_bytes=_MAX_CACHE_BYTES,
            subject="destination completion cache entry",
        )
        if existing != value:
            raise BaselineReplayImportError("destination cache entry conflicts with source")
        return
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def import_baseline_replay(
    *,
    source_roots: Iterable[Path],
    task_id: str,
    destination_cache_dir: Path,
) -> tuple[dict[str, Any], BaselineReplayReceipt]:
    """Import only cache entries referenced by one prior baseline artifact."""

    task_dir = _source_task_dir(source_roots, task_id)
    pair_path = task_dir / "pair.json"
    pair, pair_bytes = _load_json_file(
        pair_path,
        max_bytes=_MAX_PAIR_BYTES,
        subject="source pair artifact",
    )
    if not isinstance(pair, dict):
        raise BaselineReplayImportError("source pair artifact must be an object")
    baseline = pair.get("baseline")
    episode = baseline.get("episode") if isinstance(baseline, dict) else None
    audits = episode.get("completion_audits") if isinstance(episode, dict) else None
    if not isinstance(audits, list) or not audits:
        raise BaselineReplayImportError("source pair has no baseline completion audits")
    source_cache = task_dir / "producer_cache"
    try:
        cache_info = source_cache.lstat()
    except OSError as exc:
        raise BaselineReplayImportError("source producer cache is missing") from exc
    if stat.S_ISLNK(cache_info.st_mode) or not stat.S_ISDIR(cache_info.st_mode):
        raise BaselineReplayImportError("source producer cache must be a real directory")

    requests: dict[str, str] = {}
    for index, audit in enumerate(audits):
        if not isinstance(audit, dict):
            raise BaselineReplayImportError("baseline completion audit is malformed")
        request = audit.get("request_sha256")
        completion = audit.get("completion_sha256")
        if (
            not isinstance(request, str)
            or len(request) != 64
            or any(ch not in "0123456789abcdef" for ch in request)
            or not isinstance(completion, str)
            or len(completion) != 64
            or any(ch not in "0123456789abcdef" for ch in completion)
        ):
            raise BaselineReplayImportError(
                f"baseline completion audit {index} lacks SHA-256 identity"
            )
        previous = requests.setdefault(request, completion)
        if previous != completion:
            raise BaselineReplayImportError("one request maps to multiple completions")

    imported = 0
    for request, claimed_completion in sorted(requests.items()):
        source = source_cache / f"{request}.json"
        value, _ = _load_json_file(
            source,
            max_bytes=_MAX_CACHE_BYTES,
            subject="source completion cache entry",
        )
        if _canonical_sha256(value) != claimed_completion:
            raise BaselineReplayImportError(
                "source cache completion digest differs from baseline audit"
            )
        _atomic_copy(source, destination_cache_dir / source.name)
        imported += 1

    receipt = BaselineReplayReceipt(
        source_pair_path=str(pair_path),
        source_pair_sha256=hashlib.sha256(pair_bytes).hexdigest(),
        imported_entries=imported,
    )
    return pair, receipt


def alias_imported_completion(
    *,
    cache_dir: Path,
    source_request_sha256: str,
    target_request_sha256: str,
    completion_sha256: str,
) -> None:
    """Alias one audited completion when only volatile request fields drifted.

    The caller remains responsible for replaying entries in the source audit's
    exact ordinal order. This function only verifies and copies one entry; it
    never chooses a completion by semantic similarity.
    """

    if not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be pathlib.Path")
    for name, value in (
        ("source_request_sha256", source_request_sha256),
        ("target_request_sha256", target_request_sha256),
        ("completion_sha256", completion_sha256),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise BaselineReplayImportError(f"{name} must be lowercase SHA-256")
    source = cache_dir / f"{source_request_sha256}.json"
    value, _ = _load_json_file(
        source,
        max_bytes=_MAX_CACHE_BYTES,
        subject="imported source completion cache entry",
    )
    if _canonical_sha256(value) != completion_sha256:
        raise BaselineReplayImportError(
            "imported completion digest differs from source audit"
        )
    _atomic_copy(source, cache_dir / f"{target_request_sha256}.json")


__all__ = [
    "BaselineReplayImportError",
    "BaselineReplayReceipt",
    "alias_imported_completion",
    "import_baseline_replay",
]
