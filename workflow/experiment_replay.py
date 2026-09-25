"""Persist successful experiment continuations without changing default runs."""

from pathlib import Path
from typing import Any

from shadow_verifier.backends.replay import ReplayCacheBackend


class OccurrenceCacheBackend:
    def __init__(self, backend: Any, root: Path) -> None:
        self.model, self.config = backend.model, backend.config
        self.backend, self.root = backend, root
        self.index = 0
        self.last_cache_hit = False
        self.last_request_digest = None
        self.last_completion_digest = None

    def complete(self, messages, tools=None, seed=None, json_mode=False):
        cache = ReplayCacheBackend(self.backend, self.root / f"{self.index:06d}")
        completion = cache.complete(messages, tools=tools, seed=seed, json_mode=json_mode)
        self.index += 1
        self.last_cache_hit = cache.last_cache_hit
        self.last_request_digest = cache.last_request_digest
        self.last_completion_digest = cache.last_completion_digest
        return completion
