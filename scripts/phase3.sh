#!/usr/bin/env bash
# Single-container Phase-3 entrypoint.
#
#   bash scripts/phase3.sh --full          # generate + validate the placeholder run
#   bash scripts/phase3.sh --verify-only   # read-only re-verification
#
# The placeholder run is deterministic (one explicit mt19937_64 seed) and writes
# only under $ARTS/eval. The frozen Phase-2 ray set is never touched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTS_ROOT="${RAYLIDAR_ARTIFACTS_ROOT:-/data-20T/hybrid-lidar-sim/artifacts}"
ARTS="${ARTS_ROOT}/03"
EVAL_DIR="${ARTS}/eval"
CLI="$ROOT/build/${RAYLIDAR_PRESET:-linux-debug}/raylidar_cli"
CONFIG="$ROOT/configs/phase3.yaml"
PROTOCOL_SIDECAR="$ROOT/docs/protocol.sha256"
SIMIO_SIDECAR="$ROOT/docs/sim-io.sha256"

usage() {
  cat <<'EOF'
Usage: bash scripts/phase3.sh [--full|--verify-only]

  --full          eval placeholder -> eval validate -> sidecar verification
  --verify-only   eval validate -> sidecar verification (artifacts untouched)

Exit codes: 0 success, 2 usage, otherwise the failing command's exit code.
EOF
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

mode="full"
case "${1:-}" in
  --full)
    mode="full"
    shift
    ;;
  --verify-only)
    mode="verify"
    shift
    ;;
  --help | -h)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac
if [[ $# -ne 0 ]]; then
  echo "unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

cd "$ROOT"

if [[ "$mode" == "full" ]]; then
  run "$CLI" eval placeholder \
    --config "$CONFIG" \
    --out-dir "$EVAL_DIR"
fi

run "$CLI" eval validate \
  --metrics "$EVAL_DIR/metrics.json" \
  --csv "$EVAL_DIR/metrics.csv" \
  --protocol "$PROTOCOL_SIDECAR" \
  --sim-io "$SIMIO_SIDECAR"

run sha256sum -c docs/protocol.sha256 docs/sim-io.sha256
