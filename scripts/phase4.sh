#!/usr/bin/env bash
# Single Phase-4 entrypoint.
#
#   bash scripts/phase4.sh --env
#   bash scripts/phase4.sh --test [pytest args...]
#   bash scripts/phase4.sh --config-check --config <path>
#   bash scripts/phase4.sh --stage <name> --window <smoke|full> --config <path>
#   bash scripts/phase4.sh --dense-smoke --config <path>
#   bash scripts/phase4.sh --route-smoke --route <name>|all [--config <path>]
#   bash scripts/phase4.sh --smoke [--route <name>|all] --config <path>
#   bash scripts/phase4.sh --full --route <name> --config <path>
#   bash scripts/phase4.sh --verify-only --config <path> --run-id <id>
#   bash scripts/phase4.sh --promote --run-id <id> --config <path>
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
  --route-smoke               run one route (or all runnable routes, + A/B) on smoke
  --smoke [--route all]       run the route's stages on the smoke window
  --full                      run the route's stages on the full window
  --verify-only               re-hash a run and re-validate its canonical mesh
  --promote                   write the A/B report and promote a passing mesh

Options:
  --config <path>            route config YAML
  --window <name>            smoke|full (default: smoke)
  --route <name>             route override, or 'all' for --smoke
  --run-id <id>              stable run identifier (needed by --verify-only)
  --stop-before <stage>      stop before this stage (e.g. 'report')
  --allow-failing-smoke      run --full for diagnosis after a failing smoke gate
                              (records a D-09 route decision; never promotes)
  --bundle-root <p>          bundle root for --env

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
STOP_BEFORE=""
ALLOW_FAILING_SMOKE=0
BUNDLE_ROOT="$DEFAULT_BUNDLE"
PYTEST_ARGS=()

while (($#)); do
  case "$1" in
    --env | --dense-smoke | --route-smoke | --smoke | --full | --verify-only | --promote)
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
    --stop-before)
      STOP_BEFORE="${2:-}"
      shift 2
      ;;
    --allow-failing-smoke)
      ALLOW_FAILING_SMOKE=1
      shift
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

run_id_for_config_route() {
  py -c 'import sys
from tools.mesh.reconstruct.config import config_sha256, load_config
cfg = dict(load_config(sys.argv[1]))
cfg["route"] = sys.argv[2]
print(f"{sys.argv[2]}-{config_sha256(cfg)[:12]}")' "$1" "$2"
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

smoke_gate_pass() {
  local report="$1"
  py -c 'import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
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
raise SystemExit(0 if seen and all(seen) else 1)' "$report"
}

record_gate_failure() {
  local window="$1" run_id="$2" decision="$3" reason="$4"
  local report="$ARTS/runs/$run_id/$window/g3_report.json"
  [[ -f "$report" ]] || return 0
  py -c 'import sys
from tools.mesh.reconstruct import g3
path = g3.write_route_decision(
    sys.argv[1], sys.argv[2], sys.argv[3],
    decision=sys.argv[4], reason=sys.argv[5],
)
print(f"route decision: {path}")' "$ARTS" "$run_id" "$report" "$decision" "$reason"
}

record_run_failure() {
  local window="$1" run_id="$2" code="$3" status="$4"
  local run_dir="$ARTS/runs/$run_id/$window"
  [[ -d "$run_dir" ]] || return 0
  py -c 'import json, pathlib, sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1]) / "RESOURCE_LIMIT.json"
payload = {
    "schema_version": 1,
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "error": sys.argv[4],
    "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(f"run failure recorded: {path}")' "$run_dir" "$status" "$code" \
    "pipeline exited $code in the $window window; see the run log"
}

run_gate() {
  local window="$1" run_id="$2"
  shift 2
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  set +e
  "$@"
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    if [[ $code -eq 6 ]]; then
      record_gate_failure "$window" "$run_id" "gate_failed" \
        "G3 failed on the $window window; no mesh was promoted" || true
    else
      if [[ $code -eq 137 || $code -eq 9 ]]; then
        STATUS="not_run_resource_limit"
      else
        STATUS="run_failed"
      fi
      record_run_failure "$window" "$run_id" "$code" "$STATUS" || true
    fi
    echo "FAILED ($code): $*" >&2
    exit "$code"
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
  # Config paths are opened inside the container, where the repo is /src.
  printf '%s' "$(container_path "$candidate")"
}

route_smoke_preflight() {
  local name="$1"
  local cfg="$2"
  local result code message
  result="$(py -c 'import sys
from tools.mesh.reconstruct.routes import route_smoke_preflight
code, message = route_smoke_preflight(sys.argv[1], sys.argv[2], sys.argv[3])
print(code)
print(message)' "$name" "$cfg" "$ARTS")"
  code="$(printf '%s\n' "$result" | sed -n '1p')"
  message="$(printf '%s\n' "$result" | sed -n '2,$p')"
  if [[ "$code" != "0" ]]; then
    printf '%s\n' "$message" >&2
    exit "$code"
  fi
}

route_smoke_run() {
  local name="$1"
  local cfg="$2"
  local run_id="$3"
  local stages stage
  stages="$(py -c 'import sys
from tools.mesh.reconstruct.routes import route_stages
print(" ".join(route_stages(sys.argv[1])))' "$name")"
  for stage in $stages; do
    run pipeline --config "$cfg" --window smoke --route "$name" \
      --run-id "$run_id" --stage "$stage"
  done
}

fallback_backend() {
  py -c 'import sys
from tools.mesh.reconstruct.routes import fallback_state
print(fallback_state(sys.argv[1]).get("backend") or "")' "$ARTS"
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

  route-smoke)
    REQUESTED="$ROUTE"
    if [[ -z "$REQUESTED" ]]; then
      if [[ -z "$CONFIG" ]]; then
        echo "--route-smoke requires --route <name> or --config <path>" >&2
        exit 2
      fi
      REQUESTED="$(py -c 'import sys
from tools.mesh.reconstruct.config import load_config
print(load_config(sys.argv[1])["route"])' "$CONFIG")"
    fi

    if [[ "$REQUESTED" == "all" ]]; then
      CANON_ID="colmap_openmvs-smoke-4615-4624"
      O3D_ID="open3d_alpha-smoke-4615-4624"
      [[ -n "$RUN_ID" ]] && CANON_ID="$RUN_ID-colmap_openmvs"
      [[ -n "$RUN_ID" ]] && O3D_ID="$RUN_ID-open3d_alpha"

      CANON_CFG="$(route_config colmap_openmvs)"
      O3D_CFG="$(route_config open3d_alpha)"
      route_smoke_preflight colmap_openmvs "$CANON_CFG"
      route_smoke_run colmap_openmvs "$CANON_CFG" "$CANON_ID"

      FALLBACK="$(fallback_backend)"
      if [[ "$FALLBACK" == "openmvs_densify" ]]; then
        O3D_WORK="$ARTS/runs/$O3D_ID/smoke"
        run pipeline --config "$O3D_CFG" --window smoke \
          --route open3d_alpha --run-id "$O3D_ID" --stage model
        run py -c 'import sys
from tools.mesh.reconstruct.routes import seed_open3d_dense
print(seed_open3d_dense(sys.argv[1], sys.argv[2]))' \
          "$O3D_WORK" "$ARTS/runs/$CANON_ID/smoke/openmvs/scene_dense.ply"
        run pipeline --config "$O3D_CFG" --window smoke \
          --route open3d_alpha --run-id "$O3D_ID" --stage build
        SHARED=1
        NOTE="Open3D consumed the canonical run's OpenMVS scene_dense.ply (approved openmvs_densify fallback); the A/B remains a mesher comparison."
      else
        route_smoke_preflight open3d_alpha "$O3D_CFG"
        route_smoke_run open3d_alpha "$O3D_CFG" "$O3D_ID"
        SHARED=0
        NOTE="Canonical route densifies internally with OpenMVS DensifyPointCloud; Open3D consumes the COLMAP CUDA fused cloud (dense_backend=colmap_cuda). No dense-backend fallback checkpoint was selected."
      fi

      run py -c 'import sys
from tools.mesh.reconstruct.routes import write_ab_from_stages
path = write_ab_from_stages(
    sys.argv[1],
    {"colmap_openmvs": sys.argv[2], "open3d_alpha": sys.argv[3]},
    {"shared_dense_cloud": sys.argv[4] == "1", "note": sys.argv[5]},
)
print(path)' "$ARTS" "$CANON_ID" "$O3D_ID" "$SHARED" "$NOTE"
      exit 0
    fi

    if [[ -z "$CONFIG" ]]; then
      CONFIG="$(route_config "$REQUESTED")"
    fi
    RUN_ID="${RUN_ID:-${REQUESTED}-smoke-4615-4624}"
    route_smoke_preflight "$REQUESTED" "$CONFIG"
    route_smoke_run "$REQUESTED" "$CONFIG" "$RUN_ID"
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
      [[ -z "$RUN_ID" ]] && RUN_ID="$(default_run_id "$ROUTE")"
      args=(--config "$CONFIG" --window smoke --run-id "$RUN_ID")
      [[ -n "$ROUTE" ]] && args+=(--route "$ROUTE")
      [[ -n "$STOP_BEFORE" ]] && args+=(--stop-before "$STOP_BEFORE")
      run_gate smoke "$RUN_ID" pipeline "${args[@]}"
    fi
    ;;

  full)
    require_config
    [[ -z "$RUN_ID" ]] && RUN_ID="$(default_run_id "$ROUTE")"
    GATE="$ARTS/runs/$RUN_ID/smoke/g3_report.json"
    if ! smoke_gate_pass "$GATE"; then
      if [[ "$ALLOW_FAILING_SMOKE" == "1" && -f "$GATE" ]]; then
        echo "WARNING: smoke G3 failed for $RUN_ID; running the full window for" >&2
        echo "diagnosis only. Promotion still requires g3.pass == true." >&2
        record_gate_failure smoke "$RUN_ID" "diagnostic_full_run" \
          "Operator requested a full-window diagnostic after a failing smoke gate"
      else
        echo "refusing --full: no passing smoke run for $RUN_ID ($GATE)" >&2
        exit 3
      fi
    fi
    args=(--config "$CONFIG" --window full --run-id "$RUN_ID")
    [[ -n "$ROUTE" ]] && args+=(--route "$ROUTE")
    [[ -n "$STOP_BEFORE" ]] && args+=(--stop-before "$STOP_BEFORE")
    run_gate full "$RUN_ID" pipeline "${args[@]}"
    ;;

  promote)
    require_config
    if [[ -z "$RUN_ID" ]]; then
      echo "--promote requires --run-id" >&2
      exit 2
    fi
    args=(--run-id "$RUN_ID" --config "$CONFIG")
    O3D_ID="$(run_id_for_config_route "$(route_config open3d_alpha)" open3d_alpha)"
    if [[ -d "$ARTS/runs/$O3D_ID/full" ]]; then
      args+=(--ab-route "open3d_alpha=$O3D_ID")
    fi
    [[ -n "${ASSETS_ROOT:-}" ]] && args+=(--assets-root "$ASSETS_ROOT")
    run py -m tools.mesh.reconstruct.report "${args[@]}"
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
