# Mesh Reconstruction Package

This document describes the Phase 4 reconstruction package and the accompanying
data bundle. The package is self-contained under `tools/mesh/reconstruct/` and
does not enforce a hard mesh acceptance threshold.

## Routes

- **Canonical:** COLMAP known-pose model → OpenMVS
  `DensifyPointCloud → ReconstructMesh → RefineMesh`.
- **A/B:** Open3D alpha shape on the COLMAP CUDA dense cloud.

The canonical output is a W-frame, metres, triangle mesh exported as
`mesh.ply`, `mesh.glb`, and `mesh.obj`, with `mesh_manifest.json` and
`SHA256SUMS`. The viewer GLB is a separate decimated copy.

## Package layout

```text
tools/mesh/reconstruct/
  contracts.py          # MeshArtifact / RouteDescriptor / WindowSpec / MeshReconstructor
  config.py             # strict YAML config and canonical SHA-256
  env.py                # toolchain probe and environment report
  cameras.py            # P2 bundle, T_W_C / T_W_V, window helpers
  colmap_model.py       # known-pose COLMAP model
  dense.py              # COLMAP CUDA PatchMatch / fusion
  routes/               # colmap_openmvs, open3d_alpha, openmvs_direct
  meshio.py             # PLY / GLB / OBJ writers
  manifest.py           # mesh_manifest.json + SHA256SUMS
  cleanup.py            # deterministic weld, components, degenerates, decimation
  ingest.py             # route-agnostic canonical adapter
  validate.py           # C++ mesh validation wrapper
  g3.py                 # diagnostic mesh measurements
  report.py             # A/B report, promotion, index
  tests/                # package tests (root C++ tests are not part of this package)
```

## Commands

All commands run through the project container:

```bash
bash scripts/phase4.sh --env
bash scripts/phase4.sh --test
bash scripts/phase4.sh --smoke --route colmap_openmvs \
  --config tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml
bash scripts/phase4.sh --full --route colmap_openmvs \
  --config tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml
bash scripts/phase4.sh --promote --run-id <run-id> \
  --config tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml
bash scripts/phase4.sh --verify-only --run-id <run-id> \
  --config tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml
```

The C++ ingest gate is:

```bash
raylidar_cli mesh validate --manifest <asset>/mesh_manifest.json \
  --run-dir <asset> --verify-hashes
```

## Diagnostics, not a gate

`g3_report.json` contains `status: "measured"`, measured values, and reference
values. It has no `pass` or `failures` fields. The reference values are useful
for comparison but are **not enforced**; `--full`, `--promote`, and the pipeline
do not fail because a reference value is missed.

The full canonical mesh in this repository measured:

| Measurement | Value |
|---|---|
| camera-visible GT x/y ratio | 0.9016 / 1.0447 |
| z sanity band | within |
| forward hit rate | 1.0000 |
| components | 17, largest 0.9085 |
| degenerate fraction | 0.0 |
| smoke median range error | 0.2014 m |

No alignment, ICP, Umeyama, or scale correction is applied anywhere in the
construction path. GT and frozen rays are read only by diagnostics.

## Data bundle

A separate team-restricted Google Drive bundle contains:

- `mesh/mesh.ply` — canonical W-frame, metres mesh
- `rgb/image_00/` — 60 rectified RGB frames used for reconstruction (4615–4674)
- `camera/` — `cameras.json`, `calib.json`, `frames.csv`,
  `camera_params.json` (both cameras, per-frame `T_W_C`/`T_W_V`, ray convention,
  and the five camera-outside directions)
- `gt_rays/gt_rays.bin` — 1,765,175 frozen camera-origin rays, RLGR 96-byte
  records, `t_min = 1e-4 m`, `t_max = 60 m`
- manifests, README, and `SHA256SUMS`

The 60 RGB images are the reconstruction inputs; `image_01` parameters are
included but its images were not used. The bundle uses KITTI-360 data, which is
non-commercial research data; share it through a team-restricted link.

Google Drive link (team-restricted): https://drive.google.com/file/d/1NhETMB441QMgmPOMsnXYZtpdhLCfxOu7/view?usp=sharing

## Tests

```bash
bash scripts/phase4.sh --test        # 162 package tests under tools/mesh/reconstruct/tests
bash scripts/test.sh -LE gpu         # C++ build + 3 CLI smoke tests
```

The root `tests/` tree is intentionally excluded and gitignored; the
reconstruction package keeps its own tests under `tools/mesh/reconstruct/tests`
so it can be developed and run independently.

## Commit conventions

Use Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`,
`test:`). Do not put phase numbers in commit tags or messages; phase context
belongs in the issue and PR description.
