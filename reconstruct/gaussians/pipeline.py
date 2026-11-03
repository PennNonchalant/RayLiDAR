"""Stage dispatch for the Phase-5 Gaussian pipeline.

The pipeline is a table of eleven stage names.  Each stage is a lazily imported
callable that receives one context mapping::

    {
        "cfg": dict,
        "config_sha256": str,
        "method": str | None,
        "smoke": bool,
        "run_id": str,
    }

Later plans add stage modules (``gs3d``, ``gs2d``, ``probe``, ``hits_runner``,
``eval_runner``, ``report``) without editing this table, the shell entrypoint,
or the config schema.  A missing module is not an import-time error: invoking
it raises :class:`StageUnavailable` so a half-built pipeline fails loudly.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import config_sha256, load_config
from .run_utils import write_json_atomic


class StageUnavailable(RuntimeError):
    """The stage is known but its implementation module/function is absent."""


class StageContractError(RuntimeError):
    """A stage was dispatched with an unknown name or returned a bad payload."""


# stage -> "module:callable"
STAGES: dict[str, str] = {
    "dataset": "reconstruct.gaussians.dataset:run",
    "io02": "reconstruct.gaussians.io02:run",
    "train-3dgs": "reconstruct.gaussians.methods.gs3d:train",
    "render-3dgs": "reconstruct.gaussians.methods.gs3d:render",
    "train-2dgs": "reconstruct.gaussians.methods.gs2d:train",
    "render-2dgs": "reconstruct.gaussians.methods.gs2d:render",
    "probe": "reconstruct.gaussians.probe:run",
    "hits": "reconstruct.gaussians.hits_runner:run",
    "eval": "reconstruct.gaussians.eval_runner:run",
    "report": "reconstruct.gaussians.report:run",
    "verify-only": "reconstruct.gaussians.report:verify",
}

CONFIG_CHECK_STAGE = "config-check"


def load_stage(name: str) -> Callable[..., Any]:
    """Import a stage callable lazily.

    A missing module or attribute becomes :class:`StageUnavailable` so a later
    plan's unfinished stage fails with the stage name instead of an opaque
    import traceback.
    """
    if name not in STAGES:
        raise StageContractError(f"unknown stage: {name}")
    module_name, _, attribute = STAGES[name].partition(":")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise StageUnavailable(f"{name}: {exc}") from exc


def build_context(
    cfg: dict,
    method: str | None = None,
    smoke: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the single argument mapping every stage receives."""
    return {
        "cfg": cfg,
        "config_sha256": config_sha256(cfg),
        "method": method,
        "smoke": bool(smoke),
        "run_id": run_id or cfg["run_id"],
    }


def run_stage(
    name: str,
    cfg: dict,
    method: str | None = None,
    smoke: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run one stage and persist its returned mapping under ``stages/``."""
    if name not in STAGES:
        raise StageContractError(f"unknown stage: {name}")
    stage = load_stage(name)
    context = build_context(cfg, method=method, smoke=smoke, run_id=run_id)
    result = stage(context)
    if not isinstance(result, Mapping):
        raise StageContractError(f"stage {name} must return a mapping")
    payload = dict(result)
    artifacts_root = Path(cfg["paths"]["artifacts_root"])
    record_path = artifacts_root / "stages" / f"{name}.json"
    write_json_atomic(record_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reconstruct.gaussians.pipeline",
        description="Phase-5 Gaussian stage dispatcher",
    )
    parser.add_argument("--config", required=True, help="Phase-5 YAML config")
    parser.add_argument(
        "--stage",
        required=True,
        help=f"stage name, or {CONFIG_CHECK_STAGE} to print the config hash",
    )
    parser.add_argument("--method", default=None, help="optional method override")
    parser.add_argument("--smoke", action="store_true", help="smoke run")
    parser.add_argument("--run-id", default=None, help="stable run identifier")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    digest = config_sha256(cfg)
    if args.stage == CONFIG_CHECK_STAGE:
        print(f"config_sha256={digest}")
        return 0

    if args.stage not in STAGES:
        print(f"unknown stage: {args.stage}", file=sys.stderr)
        return 2

    try:
        run_stage(
            args.stage,
            cfg,
            method=args.method,
            smoke=args.smoke,
            run_id=args.run_id,
        )
    except StageUnavailable as exc:
        print(f"stage unavailable: {exc}", file=sys.stderr)
        return 3
    except (StageContractError, RuntimeError, ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"stage={args.stage} config_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
