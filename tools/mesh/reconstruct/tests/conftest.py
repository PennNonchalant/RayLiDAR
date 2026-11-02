"""Shared pytest fixtures for the Phase-4 reconstruction package."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root from this file's location.

    The test file lives at ``tools/mesh/reconstruct/tests/conftest.py``; walk
    upwards until the project root markers are found instead of hard-coding an
    index.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "CMakeLists.txt").is_file() and (candidate / "tools").is_dir():
            return candidate
    raise RuntimeError("could not locate the repository root")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import pytest  # noqa: E402  (after sys.path setup)


@pytest.fixture()
def tmp_artifacts(tmp_path: Path) -> Path:
    """An isolated artifact root under the test's temporary directory."""
    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root
