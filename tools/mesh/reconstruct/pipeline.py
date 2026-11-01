"""Stage dispatch for the Phase-4 reconstruction pipeline.

The pipeline is a table of six stages.  Each stage is a pure function of the
context keys it declares plus files under ``workdir``; there is no global
mutable state.  Later plans add stage modules, never new CLI flags.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import config_sha256, load_config
from .contracts import WindowSpec
from .routes import ROUTES, RouteUnavailable, resolve_route


class StageUnavailable(RuntimeError):
    """The stage is known but its implementation module/function is absent."""


class StageContractError(RuntimeError):
    """A stage was called without a context key or argument it requires."""


# stage -> "module:callable"
STAGES: dict[str, str] = {
    "model": "tools.mesh.reconstruct.colmap_model:run",
    "dense": "tools.mesh.reconstruct.dense:produce",
    "build": "tools.mesh.reconstruct.routes:build",
    "ingest": "tools.mesh.reconstruct.ingest:ingest",
    "gate": "tools.mesh.reconstruct.g3:evaluate",
    "report": "tools.mesh.reconstruct.report:write",
}

STAGE_ORDER: tuple[str, ...] = ("model", "dense", "build", "ingest", "gate", "report")

# Which context key receives the stage's own result payload.
STAGE_CONTEXT_KEYS: dict[str, str | None] = {
    "model": "colmap_result",
    "dense": "dense_result",
    "build": "route_result",
    "ingest": "artifact",
    "gate": "g3_report",
    "report": None,
}

# Exactly the keys every stage may declare.
CONTEXT_KEYS: tuple[str, ...] = (
    "bundle_root",
    "window_name",
    "frames",
    "cfg",
    "workdir",
    "env",
    "config_sha256",
    "colmap_result",
    "dense_result",
    "route_result",
    "artifact",
    "g3_report",
    "routes",
)


def load_stage(name: str) -> Callable[..., Any]:
    """Import a stage callable lazily.

    A missing module or function is reported as :class:`StageUnavailable` so a
    partially implemented pipeline fails loudly instead of silently skipping.
    """
    if name not in STAGES:
        raise StageContractError(f"unknown stage: {name}")
    module_name, _, attribute = STAGES[name].partition(":")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise StageUnavailable(f"{name}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_stage(name: str, ctx: dict) -> Any:
    """Run one stage against ``ctx`` and merge its result back into ``ctx``."""
    if not ctx.get("config_sha256"):
        raise StageContractError("run_stage: config_sha256 is required in the context")

    stage = load_stage(name)
    signature = inspect.signature(stage)
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.name in ctx:
            kwargs[parameter.name] = ctx[parameter.name]
        elif parameter.default is inspect.Parameter.empty:
            raise StageContractError(
                f"stage {name}: missing context key '{parameter.name}'"
            )

    result = stage(**kwargs)

    if isinstance(result, dict):
        workdir = ctx.get("workdir")
        if workdir is not None:
            _write_json(Path(workdir) / "stages" / f"{name}.json", result)
        context_key = STAGE_CONTEXT_KEYS.get(name)
        if context_key and context_key in result:
            ctx[context_key] = result[context_key]
        if name == "build" and isinstance(result.get("routes"), list):
            routes = ctx.setdefault("routes", [])
            routes.extend(result["routes"])

    return result


def _load_previous_stages(ctx: dict, workdir: Path) -> None:
    for stage, context_key in STAGE_CONTEXT_KEYS.items():
        path = workdir / "stages" / f"{stage}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StageContractError(f"corrupt stage record {path}: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        if context_key and context_key in payload:
            ctx[context_key] = payload[context_key]
        if stage == "build" and isinstance(payload.get("routes"), list):
            ctx["routes"] = list(payload["routes"])


def _default_workdir(cfg: dict, run_id: str, window: str) -> Path:
    return Path(cfg["out_root"]) / "runs" / run_id / window


def _build_context(
    cfg: dict, window: str, workdir: Path, run_id: str, env: dict
) -> dict:
    frames = tuple(cfg["windows"][window])
    WindowSpec(window, frames)
    return {
        "bundle_root": Path(cfg["bundle_root"]),
        "window_name": window,
        "frames": frames,
        "cfg": cfg,
        "workdir": workdir,
        "env": env,
        "config_sha256": config_sha256(cfg),
        "colmap_result": None,
        "dense_result": None,
        "route_result": None,
        "artifact": None,
        "g3_report": None,
        "routes": [],
    }


def _load_env_report(cfg: dict) -> dict:
    path = Path(cfg["out_root"]) / "env_report.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.mesh.reconstruct.pipeline",
        description="Phase-4 reconstruction stage dispatcher",
    )
    parser.add_argument("--config", required=True, help="route config YAML")
    parser.add_argument("--stage", help="run exactly this stage")
    parser.add_argument("--window", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--route", help="override the config route for this run")
    parser.add_argument("--workdir", help="run workspace (default: out_root/runs/...)")
    parser.add_argument("--run-id", help="stable run identifier")
    parser.add_argument(
        "--stop-before", help="run all stages up to (not including) this stage"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.route:
            if args.route not in ROUTES:
                raise RouteUnavailable(f"unknown route: {args.route}")
            cfg = dict(cfg)
            cfg["route"] = args.route

        if args.stage == "config-check":
            print(config_sha256(cfg))
            return 0

        run_id = args.run_id or f"{cfg['route']}-{config_sha256(cfg)[:12]}"
        workdir = (
            Path(args.workdir)
            if args.workdir
            else _default_workdir(cfg, run_id, args.window)
        )
        ctx = _build_context(cfg, args.window, workdir, run_id, _load_env_report(cfg))
        _load_previous_stages(ctx, workdir)

        if args.stop_before:
            if args.stop_before not in STAGES:
                raise StageContractError(f"unknown stage: {args.stop_before}")
            for stage in STAGE_ORDER:
                if stage == args.stop_before:
                    break
                run_stage(stage, ctx)
            return 0

        if args.stage:
            run_stage(args.stage, ctx)
            return 0

        for stage in STAGE_ORDER:
            run_stage(stage, ctx)
        return 0
    except (
        StageUnavailable,
        StageContractError,
        RouteUnavailable,
        ValueError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
