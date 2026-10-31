#!/usr/bin/env bash
# Single-container Phase-2 entrypoint.
#
#   bash scripts/phase2.sh --full          # run the whole pipeline
#   bash scripts/phase2.sh --verify-only   # read-only re-verification
#
# Every stage takes explicit paths and the script stops on the first non-zero
# exit. There is no eval and no user-supplied shell fragment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTS_ROOT="${RAYLIDAR_ARTIFACTS_ROOT:-/data-20T/hybrid-lidar-sim/artifacts}"
ARTS="${ARTS_ROOT}/02"
PHASE1="${ARTS_ROOT}/01"
CLI="$ROOT/build/${RAYLIDAR_PRESET:-linux-debug}/raylidar_cli"
CONFIG="$ROOT/configs/phase2.yaml"
PROTOCOL="$ROOT/configs/protocol.yaml"
DECISIONS="$ROOT/docs/decisions.md"
SIDECAR="$PHASE1/decisions.sha256"

usage() {
  cat <<'EOF'
Usage: bash scripts/phase2.sh [--full|--verify-only]

  --full          prereg verify -> frame index -> selection -> mask build ->
                  60-frame uncurl -> GT cache -> rays -> waterfall/freeze/verify
                  -> g2-check
  --verify-only   prereg verify -> rays-verify verify -> g2-check

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

run "$CLI" prereg verify \
  --decisions "$DECISIONS" \
  --protocol "$PROTOCOL" \
  --sidecar "$SIDECAR"

if [[ "$mode" == "verify" ]]; then
  run "$CLI" rays-verify verify \
    --rays-manifest "$ARTS/rays/rays_manifest.json" \
    --freeze "$ARTS/rays/RAY_SET_FREEZE.json"
  run "$CLI" g2-check \
    --config "$CONFIG" \
    --artifacts-dir "$ARTS" \
    --out "$ARTS/g2/G2_REPORT"
  exit 0
fi

run "$CLI" frame-index \
  --config "$CONFIG" \
  --out-dir "$ARTS/streams"

run "$CLI" segment-select \
  --config "$CONFIG" \
  --frame-index "$ARTS/streams/frame_index.json" \
  --out-dir "$ARTS/segment"

# mask-build must run before uncurl-check: the uncurl diagnostic has no
# unmasked fallback for real frames.
run "$CLI" mask-build \
  --config "$CONFIG" \
  --frame-list "$ARTS/segment/segment_manifest.json" \
  --out-dir "$ARTS/masks"

run "$CLI" uncurl-check \
  --config "$CONFIG" \
  --frame-list "$ARTS/segment/segment_manifest.json" \
  --mask-dir "$ARTS/masks" \
  --out-dir "$ARTS/uncurl"

run "$CLI" gt-cache \
  --config "$CONFIG" \
  --segment-manifest "$ARTS/segment/segment_manifest.json" \
  --out-dir "$ARTS/gt"

run "$CLI" rays-build \
  --config "$CONFIG" \
  --gt-manifest "$ARTS/gt/gt_manifest.json" \
  --out-dir "$ARTS/rays"

run "$CLI" rays-verify waterfall \
  --config "$CONFIG" \
  --rays-manifest "$ARTS/rays/rays_manifest.json" \
  --mask-manifest "$ARTS/masks/mask_manifest.json" \
  --gt-manifest "$ARTS/gt/gt_manifest.json"

run "$CLI" rays-verify freeze \
  --config "$CONFIG" \
  --rays-manifest "$ARTS/rays/rays_manifest.json" \
  --freeze "$ARTS/rays/RAY_SET_FREEZE.json"

run "$CLI" rays-verify verify \
  --rays-manifest "$ARTS/rays/rays_manifest.json" \
  --freeze "$ARTS/rays/RAY_SET_FREEZE.json"

run "$CLI" g2-check \
  --config "$CONFIG" \
  --artifacts-dir "$ARTS" \
  --out "$ARTS/g2/G2_REPORT"
