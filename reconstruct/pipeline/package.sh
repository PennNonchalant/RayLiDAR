#!/usr/bin/env bash
# Zip the handoff bundle for sharing.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BUNDLE="${1:-$BUNDLE_ROOT}"
if [[ ! -d "$BUNDLE" ]]; then
  echo "missing bundle: $BUNDLE" >&2
  exit 2
fi
ZIP="${BUNDLE%/}.zip"
"$PYTHON" - "$BUNDLE" "$ZIP" <<'PY'
import sys, zipfile
from pathlib import Path
root = Path(sys.argv[1]); zip_path = Path(sys.argv[2])
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root.parent))
print(f"wrote {zip_path}")
PY
