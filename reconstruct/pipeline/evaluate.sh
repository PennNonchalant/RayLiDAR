#!/usr/bin/env bash
# Reconstruction + LiDAR-simulation quality evaluation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" -m reconstruct.gaussians.pipeline --config "$GAUSSIAN_CONFIG" --stage eval "$@"
