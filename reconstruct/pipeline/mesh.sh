#!/usr/bin/env bash
# Multi-route mesh reconstruction (COLMAP/OpenMVS/Open3D).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" -m reconstruct.mesh --config "$MESH_CONFIG" "$@"
