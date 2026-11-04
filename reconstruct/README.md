# reconstruct

Self-contained KITTI-360 reconstruction and evaluation pipeline. The package is
pure Python (plus external reconstruction binaries such as COLMAP, OpenMVS,
COLMAP, OpenMVS, and the official 3DGS/2DGS code) and is independent of the C++ Vulkan
runtime in `src/`.

```
reconstruct/
├── data_preprocess/   # KITTI-360 -> frozen artifacts + handoff bundle (Python)
├── mesh/              # COLMAP -> OpenMVS -> canonical PLY/GLB/OBJ
├── gaussians/         # 3DGS vs 2DGS training, rendering, hits, report
├── eval/              # Phase-3 metrics (range, miss, coverage, P/R/F, Chamfer)
├── pipeline/          # full preprocess -> reconstruct -> evaluate -> package scripts
└── test/              # local end-to-end checks (gitignored, never committed)
```

## Quick start

```bash
export PYTHONPATH="$PWD"
python -m reconstruct.data_preprocess --help
python -m reconstruct.mesh --help
python -m reconstruct.gaussians.pipeline --help
python -m reconstruct.eval --help

bash reconstruct/pipeline/run_all.sh
```


## Test segments

Two windows are configured and selected entirely from config files:

| Segment | data_preprocess | mesh | gaussians |
|---|---|---|---|
| 1820-1879 | `configs/preprocess_1820.yaml` | `configs/mesh_1820.yaml` | `configs/gaussians_1820.yaml` |
| 7426-7485 | `configs/preprocess_7426.yaml` | `configs/mesh_7426.yaml` | `configs/gaussians_7426.yaml` |
| 4615-4674 | `configs/preprocess_4615.yaml` | `configs/mesh_4615.yaml` | `configs/gaussians_4615.yaml` |

Select a segment by pointing the pipeline scripts at the matching configs:

```bash
PHASE2_CONFIG=reconstruct/configs/preprocess_7426.yaml \
MESH_CONFIG=reconstruct/configs/mesh_7426.yaml \
GAUSSIAN_CONFIG=reconstruct/configs/gaussians_7426.yaml \
bash reconstruct/pipeline/run_all.sh
```

No segment is hard-coded in Python; the configs supply the frame window, bundle,
COLMAP model, artifact root, smoke frames, and probe frame.

## Data preprocessing

`reconstruct.data_preprocess` implements the offline pipeline in Python. It writes
the frozen Phase-2 artifacts (`streams`, `segment`, `masks`, `uncurl`, `gt`,
`rays`) and assembles the self-contained handoff bundle consumed by the mesh and
Gaussian stages.

```bash
python -m reconstruct.data_preprocess all \
  --config reconstruct/configs/preprocess_1820.yaml \
  --phase-dir /data-20T/hybrid-lidar-sim/artifacts/02
```

Frozen binary layouts are preserved: `RLGT` (GT cache), `RLRY` (camera rays),
`RLGR` (`gt_rays.bin`), `RLHT` (`hits.bin`), and the Phase-2 mask bit ABI.

## Evaluation

`reconstruct.eval` writes the Phase-3 `metrics.json` / `metrics.csv` schema used
by the Gaussian comparison report:

```bash
python -m reconstruct.eval real \
  --gt-rays <gt_rays.bin> --hits <hits.bin> --out-dir <dir> --mode gaussian_depth_all \
  --run-config <gaussians_1820.yaml> --protocol-config reconstruct/configs/eval.yaml \
  --protocol-doc reconstruct/configs/eval_protocol.md --model-sha256 <hex> \
  --scene-generator gaussian_3dgs --seed <n> --provenance <provenance.json>
```

## Tracking

`reconstruct/test/` is deliberately untracked (see `reconstruct/.gitignore`);
local end-to-end scripts and their outputs never enter the repository.
