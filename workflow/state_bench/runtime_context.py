"""Safe construction boundary for STATE-Bench custom agents."""

from __future__ import annotations

from typing import Any


def build_sanitized_runtime_context(
    *,
    episode_id: str,
    user_id: str,
    domain: str,
    now: str,
) -> Any:
    """Construct an official context from an explicit non-oracle allowlist."""

    from state_bench.agents.base import AgentRuntimeContext

    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be a non-empty opaque string")
    return AgentRuntimeContext(
        task_id=episode_id,
        user_id=user_id,
        domain=domain,
        now=now,
        # Host bookkeeping can encode task order or filesystem identity.  It
        # stays outside the model-visible runtime context.
        output_dir=None,
        run_idx=None,
        task_summary=None,
        state_requirements=[],
        task_requirements=[],
        config={},
    )


def sanitize_agent_runtime_context(
    context: Any,
    *,
    episode_id: str,
) -> Any:
    """Copy official runtime metadata while removing every task oracle.

    The official custom-agent API includes ``task_summary``, exact state
    requirements, non-state requirements, and a free-form config on the
    object handed to an agent.  The default agent does not prompt with them,
    but a workflow wrapper can leak them accidentally.  Import lazily so the
    workflow remains importable until the pinned upstream is on ``PYTHONPATH``.
    """

    from state_bench.agents.base import AgentRuntimeContext

    if not isinstance(context, AgentRuntimeContext):
        raise TypeError("context must be state_bench.agents.base.AgentRuntimeContext")
    return build_sanitized_runtime_context(
        episode_id=episode_id,
        user_id=context.user_id,
        domain=context.domain,
        now=context.now,
    )
