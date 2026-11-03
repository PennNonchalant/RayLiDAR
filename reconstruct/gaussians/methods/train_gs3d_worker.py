"""Seeded launcher for the pinned official 3DGS ``train.py``.

The launcher must run with its working directory set to the pinned
``gaussian-splatting`` checkout so ``train.py`` can import ``scene`` and
``gaussian_renderer`` from the checkout, exactly like the upstream script.

Upstream ``utils.general_utils.safe_state`` reseeds Python/NumPy/Torch to ``0``
before training starts.  Seeding only before ``runpy`` would therefore record a
seed that is not the effective seed.  The launcher wraps ``safe_state`` so the
official call still silences stdout but the configured seed is applied
afterwards.  Upstream code itself is never edited.

Usage::

    train_gs3d_worker.py <seed> <train.py arguments...>
"""

from __future__ import annotations

import os
import random
import runpy
import sys

import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _patch_official_safe_state(seed: int) -> None:
    """Re-seed after the official ``safe_state`` reset, keeping its stdout shim."""
    try:
        import utils.general_utils as general_utils
    except Exception:  # pragma: no cover - checkout layout is verified upstream
        return
    original = getattr(general_utils, "safe_state", None)
    if original is None:
        return

    def seeded_safe_state(silent: bool) -> None:
        original(silent)
        _seed_everything(seed)

    general_utils.safe_state = seeded_safe_state


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit("usage: train_gs3d_worker.py <seed> [train.py args...]")
    seed = int(arguments[0])
    sys.argv = ["train.py", *arguments[1:]]

    # cwd is the pinned checkout; make its packages importable even when the
    # parent process did not export PYTHONPATH.
    sys.path.insert(0, os.getcwd())
    _seed_everything(seed)
    _patch_official_safe_state(seed)
    runpy.run_path("train.py", run_name="__main__")


if __name__ == "__main__":
    main()
