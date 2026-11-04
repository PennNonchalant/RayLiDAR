#!/usr/bin/env bash
# Full reproduce flow: preprocess -> mesh -> gaussians -> evaluate -> package.
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
bash "$HERE/preprocess.sh" "$@"
bash "$HERE/mesh.sh" --window full
bash "$HERE/gaussians.sh" --stage dataset
bash "$HERE/gaussians.sh" --stage io02
bash "$HERE/gaussians.sh" --stage train-3dgs
bash "$HERE/gaussians.sh" --stage render-3dgs
bash "$HERE/gaussians.sh" --stage train-2dgs
bash "$HERE/gaussians.sh" --stage render-2dgs
bash "$HERE/gaussians.sh" --stage probe
bash "$HERE/gaussians.sh" --stage hits
bash "$HERE/evaluate.sh"
bash "$HERE/gaussians.sh" --stage report
bash "$HERE/package.sh"
