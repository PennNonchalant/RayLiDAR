#!/usr/bin/env bash
# Configure (if needed) and build inside the container.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env.container" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.container"
  set +a
fi
: "${RAYLIDAR_PRESET:=linux-debug}"
"$ROOT/scripts/configure.sh"
exec "$ROOT/scripts/run-in-container.sh" cmake --build --preset "$RAYLIDAR_PRESET" "$@"
