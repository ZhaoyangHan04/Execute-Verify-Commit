"""Provider-neutral completion values and protocols.

Concrete providers live in explicit submodules such as ``backends.dashscope``
so importing this package does not select or initialize a provider.
"""

from .model import Completion, CompletionProvenance, ToolCall, Usage
from .protocols import CompletionBackend

__all__ = [
    "Completion",
    "CompletionBackend",
    "CompletionProvenance",
    "ToolCall",
    "Usage",
]
