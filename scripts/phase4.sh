#!/usr/bin/env bash
# Single Phase-4 entrypoint.
#
#   bash scripts/phase4.sh --env
#   bash scripts/phase4.sh --test [pytest args...]
#   bash scripts/phase4.sh --config-check --config <path>
#   bash scripts/phase4.sh --stage <name> --window <smoke|full> --config <path>
#   bash scripts/phase4.sh --dense-smoke --config <path>
#   bash scripts/phase4.sh --smoke [--route <name>|all] --config <path>
#   bash scripts/phase4.sh --full --route <name> --config <path>
#   bash scripts/phase4.sh --verify-only --config <path> --run-id <id>
#
# Every Python/CLI invocation runs inside the approved project container via
# scripts/run-in-container.sh; the host only parses modes and paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY=/data-20T/hybrid-lidar-sim/envs/mesh-venv/bin/python
ASSETS=/data-20T/hybrid-lidar-sim/assets/mesh
DEFAULT_BUNDLE=/data-20T/hybrid-lidar-sim/exports/p2-60f-step10-4615-4674
CONFIGS="$ROOT/tools/mesh/reconstruct/configs"

usage() {
  cat <<'EOF'
Usage: bash scripts/phase4.sh <mode> [options]

Modes:
  --env                       probe the container toolchain, write env_report.json
  --test [pytest args...]     run the reconstruction pytest suite
  --config-check              load + hash a config (requires --config)
  --stage <name>              dispatch one pipeline stage
  --dense-smoke               run model + dense on the smoke window
  --smoke [--route all]       run all six stages on the smoke window
  --full                      run all six stages on the full window
  --verify-only               re-hash a run and re-validate its canonical mesh

Options:
  --config <path>   route config YAML
  --window <name>   smoke|full (default: smoke)
  --route <name>    route override, or 'all' for --smoke
  --run-id <id>     stable run identifier (needed by --verify-only)
  --bundle-root <p> bundle root for --env

Exit codes: 0 success, 2 usage, 3 missing precondition, otherwise the failing
command's exit code.
EOF
}

MODE=""
CONFIG=""
WINDOW="smoke"
ROUTE=""
RUN_ID=""
STAGE=""
BUNDLE_ROOT="$DEFAULT_BUNDLE"
PYTEST_ARGS=()

while (($#)); do
  case "$1" in
    --env | --dense-smoke | --smoke | --full | --verify-only)
      MODE="${1#--}"
      shift
      ;;
    --test)
      MODE="test"
      shift
      PYTEST_ARGS=("$@")
      break
      ;;
    --config-check)
      MODE="config-check"
      shift
      ;;
    --stage)
      MODE="stage"
      STAGE="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --window)
      WINDOW="${2:-}"
      shift 2
      ;;
    --route)
      ROUTE="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --bundle-root)
      BUNDLE_ROOT="${2:-}"
      shift 2
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "missing mode" >&2
  usage >&2
  exit 2
fi

case "$WINDOW" in
  smoke | full) ;;
  *)
    echo "invalid --window: $WINDOW" >&2
    exit 2
    ;;
esac

threads() {
  local value
  value=$(python3 -c 'import json, sys
try:
    print(json.load(open(sys.argv[1]))["threads"])
except Exception:
    print(8)' "$1" 2>/dev/null || true)
  printf '%s' "${value:-8}"
}

py() {
  bash "$ROOT/scripts/run-in-container.sh" env \
    PYTHONPATH=/src \
    OMP_NUM_THREADS="$THREADS" \
    OPENBLAS_NUM_THREADS="$THREADS" \
    "$VENV_PY" "$@"
}

pipeline() {
  py -m tools.mesh.reconstruct.pipeline "$@"
}

config_out_root() {
  py -c 'import sys
from tools.mesh.reconstruct.config import load_config
print(load_config(sys.argv[1])["out_root"])' "$1"
}

default_run_id() {
  py -c 'import sys
from tools.mesh.reconstruct.config import config_sha256, load_config
cfg = load_config(sys.argv[1])
route = sys.argv[2] or cfg["route"]
cfg = dict(cfg)
cfg["route"] = route
print(f"{route}-{config_sha256(cfg)[:12]}")' "$CONFIG" "${1:-}"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@" || {
    local code=$?
    echo "FAILED ($code): $*" >&2
    exit "$code"
  }
}

require_config() {
  if [[ -z "$CONFIG" ]]; then
    echo "mode '$MODE' requires --config <path>" >&2
    exit 2
  fi
}

container_path() {
  if [[ "$1" == "$ROOT"/* ]]; then
    printf '/src/%s' "${1#"$ROOT"/}"
  else
    printf '%s' "$1"
  fi
}

route_config() {
  local name="$1"
  local candidate="$CONFIGS/$name.yaml"
  if [[ "$name" == "colmap_openmvs" ]]; then
    candidate="$CONFIGS/canonical_colmap_openmvs.yaml"
  fi
  if [[ ! -f "$candidate" ]]; then
    echo "unknown route: $name" >&2
    exit 2
  fi
  printf '%s' "$candidate"
}

THREADS="${OMP_NUM_THREADS:-8}"
if [[ -n "${RAYLIDAR_ARTIFACTS_ROOT:-}" ]]; then
  ARTS="${RAYLIDAR_ARTIFACTS_ROOT%/}/04"
elif [[ -n "$CONFIG" ]]; then
  ARTS="$(config_out_root "$CONFIG")"
else
  ARTS=/data-20T/hybrid-lidar-sim/artifacts/04
fi
THREADS="$(threads "$ARTS/env_report.json")"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$ROOT"

case "$MODE" in
  env)
    run py -m tools.mesh.reconstruct.env \
      --bundle-root "$BUNDLE_ROOT" \
      --out "$ARTS/env_report.json"
    ;;

  test)
    if ((${#PYTEST_ARGS[@]} == 0)); then
      run py -m pytest tools/mesh/reconstruct/tests
    else
      run py -m pytest "${PYTEST_ARGS[@]}"
    fi
    ;;

  config-check)
    require_config
    run pipeline --config "$CONFIG" --stage config-check --window "$WINDOW"
    ;;

  stage)
    require_config
    if [[ -z "$STAGE" ]]; then
      echo "--stage requires a stage name" >&2
      exit 2
    fi
    args=(--config "$CONFIG" --stage "$STAGE" --window "$WINDOW")
    [[ -n "$ROUTE" ]] && args+=(--route "$ROUTE")
    [[ -n "$RUN_ID" ]] && args+=(--run-id "$RUN_ID")
    run pipeline "${args[@]}"
    ;;

  dense-smoke)
    require_config
    args=(--config "$CONFIG" --window smoke --stop-before build)
    [[ -n "$ROUTE" ]] && args+=(--route "$ROUTE")
    [[ -n "$RUN_ID" ]] && args+=(--run-id "$RUN_ID")
    run pipeline "${args[@]}"
    ;;

  smoke)
    if [[ "$ROUTE" == "all" ]]; then
      mapfile -t route_names < <(py -c 'from tools.mesh.reconstruct.routes import ROUTES
print("\n".join(sorted(ROUTES)))')
      for name in "${route_names[@]}"; do
        echo "== smoke: $name =="
        run pipeline --config "$(route_config "$name")" --window smoke --route "$name"
      done
    else
      require_config
      args=(--config "$CONFIG" --window smoke)
      [[ -n "$ROUTE" ]] && args+=(--route "$ROUTE")
      [[ -n "$RUN_ID" ]] && args+=(--run-id "$RUN_ID")
      run pipeline "${args[@]}"
    fi
    ;;

  full)
    require_config
    [[ -z "$RUN_ID" ]] && RUN_ID="$(default_run_id "$ROUTE")"
    GATE="$ARTS/runs/$RUN_ID/smoke/stages/gate.json"
    if ! py -c 'import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    print(f"missing smoke gate record: {path}", file=sys.stderr)
    raise SystemExit(1)
seen = []

def walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("pass", "passed"):
                seen.append(value is True)
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)

walk(json.load(path.open()))
raise SystemExit(0 if seen and all(seen) else 1)' "$GATE"; then
      echo "refusing --full: no passing smoke run for $RUN_ID ($GATE)" >&2
      exit 3
    fi
    args=(--config "$CONFIG" --window full --run-id "$RUN_ID")
    [[ -n "$ROUTE" ]] && args+=(--route "$ROUTE")
    run pipeline "${args[@]}"
    ;;

  verify-only)
    require_config
    if [[ -z "$RUN_ID" ]]; then
      echo "--verify-only requires --run-id" >&2
      exit 2
    fi
    MANIFEST="$ASSETS/$RUN_ID/mesh_manifest.json"
    [[ -f "$MANIFEST" ]] || MANIFEST="$ARTS/runs/$RUN_ID/mesh_manifest.json"
    if [[ ! -f "$MANIFEST" ]]; then
      echo "manifest not found for run $RUN_ID" >&2
      exit 3
    fi
    MESH="$(py -c 'import hashlib, json, pathlib, sys
manifest = pathlib.Path(sys.argv[1])
root = manifest.parent
found = []

def walk(node):
    if isinstance(node, dict):
        path = node.get("path")
        digest = node.get("sha256")
        if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
            found.append((path, digest))
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)

walk(json.load(manifest.open()))
if not found:
    raise SystemExit("verify: manifest has no hashed artifacts")
bad = []
ply = None
for relative, digest in found:
    candidate = pathlib.Path(relative)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.is_file():
        bad.append(f"missing: {candidate}")
        continue
    sha = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    if sha.hexdigest() != digest:
        bad.append(f"hash mismatch: {candidate}")
    if candidate.suffix.lower() == ".ply" and ply is None:
        ply = candidate
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit(1)
print(f"verified {len(found)} artifacts", file=sys.stderr)
print(ply if ply is not None else "")' "$MANIFEST")"
    CLI="build/${RAYLIDAR_PRESET:-linux-debug}/raylidar_cli"
    if [[ -n "$MESH" ]]; then
      run bash "$ROOT/scripts/run-in-container.sh" "$CLI" mesh validate \
        --mesh "$(container_path "$MESH")"
    else
      echo "verify: no canonical PLY recorded in $MANIFEST" >&2
      exit 3
    fi
    ;;

  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
