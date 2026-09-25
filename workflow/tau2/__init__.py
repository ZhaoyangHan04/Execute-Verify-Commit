"""Official τ² workflow primitives layered on ``shadow_verifier``."""

from shadow_verifier.projections import structural_diff

from .environment import Tau2ForkExecutor
from .evidence import (
    Tau2EvidenceContext,
    build_tau2_evidence,
    project_public_history,
    project_tool_result,
    snapshot_environment,
)
from .model import Tau2ToolBatch, project_tool_call
from .trajectory import canonicalize_messages

__all__ = [
    "Tau2EvidenceContext",
    "Tau2ForkExecutor",
    "Tau2ToolBatch",
    "build_tau2_evidence",
    "canonicalize_messages",
    "project_public_history",
    "project_tool_call",
    "project_tool_result",
    "snapshot_environment",
    "structural_diff",
]
