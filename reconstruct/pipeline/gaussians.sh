#!/usr/bin/env bash
# 3DGS vs 2DGS reconstruction and evaluation (Stage 5).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" -m reconstruct.gaussians.pipeline --config "$GAUSSIAN_CONFIG" "$@"
