#!/usr/bin/env bash
# Configure the project inside the container with the approved preset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env.container" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.container"
  set +a
fi
: "${RAYLIDAR_PRESET:=linux-debug}"
exec "$ROOT/scripts/run-in-container.sh" cmake --preset "$RAYLIDAR_PRESET" "$@"
