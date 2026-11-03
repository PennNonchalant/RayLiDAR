#!/usr/bin/env bash
# KITTI-360 -> frozen artifacts + handoff bundle (Python data_preprocess).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" -m reconstruct.data_preprocess all \
  --config "$PHASE2_CONFIG" \
  --phase-dir "${PHASE2_DIR:-$ARTIFACTS_ROOT/02}" \
  "$@"
