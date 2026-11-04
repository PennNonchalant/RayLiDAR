#!/usr/bin/env bash
# Shared environment for the reconstruct pipeline entrypoints.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Override these three to switch segments (for example the *_7426.yaml files).
PHASE2_CONFIG="${PHASE2_CONFIG:-$REPO_ROOT/reconstruct/configs/preprocess_1820.yaml}"
MESH_CONFIG="${MESH_CONFIG:-$REPO_ROOT/reconstruct/configs/mesh_1820.yaml}"
GAUSSIAN_CONFIG="${GAUSSIAN_CONFIG:-$REPO_ROOT/reconstruct/configs/gaussians_1820.yaml}"
ARTIFACTS_ROOT="${RAYLIDAR_ARTIFACTS_ROOT:-/data-20T/hybrid-lidar-sim/artifacts}"
BUNDLE_ROOT="${RECONSTRUCT_BUNDLE:-/data-20T/hybrid-lidar-sim/exports/reconstruct-bundle}"
