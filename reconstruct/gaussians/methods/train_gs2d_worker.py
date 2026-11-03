"""Seeded, headless launcher for the pinned official 2DGS ``train.py``.

The launcher must run with its working directory set to the pinned
``2d-gaussian-splatting`` checkout so ``train.py`` can import ``scene`` and
``gaussian_renderer`` from the checkout, exactly like the upstream script.

Two upstream behaviours have to be neutralised without editing upstream code:

* ``train.py`` calls ``network_gui.init`` unconditionally and then polls the
  GUI socket; the Phase-5 run is headless, so ``init``/``try_connect``/
  ``receive`` are replaced with inert no-ops and ``conn`` is forced to
  ``None``.
* ``utils.general_utils.safe_state`` reseeds Python/NumPy/Torch to ``0``
  before training starts.  Seeding only before ``runpy`` would therefore
  record a seed that is not the effective seed, so the launcher wraps
  ``safe_state`` and re-applies the configured seed afterwards.

Usage::

    train_gs2d_worker.py <seed> <train.py arguments...>
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


def _disable_network_gui() -> None:
    """Make the official GUI module inert before ``train.py`` imports it."""
    import gaussian_renderer.network_gui as network_gui

    network_gui.init = lambda *arguments, **keywords: None
    network_gui.try_connect = lambda *arguments, **keywords: None
    network_gui.receive = lambda *arguments, **keywords: (None, True, False, 1.0, "RGB")
    network_gui.conn = None
    network_gui.addr = None


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
        raise SystemExit("usage: train_gs2d_worker.py <seed> [train.py args...]")
    seed = int(arguments[0])
    sys.argv = ["train.py", *arguments[1:]]

    # cwd is the pinned checkout; make its packages importable even when the
    # parent process did not export PYTHONPATH.
    sys.path.insert(0, os.getcwd())
    _seed_everything(seed)
    _disable_network_gui()
    _patch_official_safe_state(seed)
    runpy.run_path("train.py", run_name="__main__")


if __name__ == "__main__":
    main()
