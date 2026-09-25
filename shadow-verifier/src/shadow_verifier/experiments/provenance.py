"""Source-control provenance helpers for reproducible workflows."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def require_clean_git_tree(path: Path, *, subject: str = "git worktree") -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot verify {subject}")
    if completed.stdout:
        raise RuntimeError(f"{subject} must be clean")
