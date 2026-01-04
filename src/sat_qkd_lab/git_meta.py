"""Git metadata helpers."""

from __future__ import annotations

from typing import Optional
import subprocess


def get_git_commit() -> Optional[str]:
    """Return git commit hash or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    commit = result.stdout.strip()
    return commit or None
