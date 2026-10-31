#!/usr/bin/env bash
# Single entrypoint for every build/test/job: run one command inside the project
# container described by .env.container.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$ROOT/.env.container" ]]; then
  echo "ERROR: $ROOT/.env.container is missing." >&2
  echo "Create it from .env.container.example and set RAYLIDAR_CONTAINER_RUN_CMD." >&2
  exit 78
fi
set -a
# shellcheck disable=SC1091
source "$ROOT/.env.container"
set +a
: "${RAYLIDAR_CONTAINER_RUN_CMD:?RAYLIDAR_CONTAINER_RUN_CMD is not set}"
: "${RAYLIDAR_CONTAINER_WORKDIR:=/workspace}"
: "${RAYLIDAR_PRESET:=linux-debug}"
: "${RAYLIDAR_ARTIFACTS_ROOT:=/data-20T/hybrid-lidar-sim/artifacts}"
export RAYLIDAR_PRESET RAYLIDAR_ARTIFACTS_ROOT
read -r -a CMD <<< "$RAYLIDAR_CONTAINER_RUN_CMD"
exec "${CMD[@]}" "$@"
