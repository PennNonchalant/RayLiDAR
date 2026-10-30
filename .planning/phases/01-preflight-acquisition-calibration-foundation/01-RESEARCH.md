# Phase 1 — Research: Preflight, Acquisition & Calibration Foundation

**Researched:** 2026-10-30
**Question:** What do we need to know to plan Phase 1 well?
**Requirements in scope:** DATA-01, DATA-02, DATA-04, DATA-09, EVAL-15
**Gate:** G1 calibration gate plus explicit dataset-readiness sign-off.

---

## Executive Summary

1. **The official KITTI-360 download is login-gated.** `https://www.cvlibs.net/datasets/kitti-360/user_login.php` requires a registered account; the download page is not a public bucket index. Acquisition therefore contains a genuine human action (register/login, accept the non-commercial terms, download the archives). The agent can automate extraction, validation, hashing, and installation, but must not invent download URLs or bypass the licence.
2. **The local `/data-20T/kitti` tree is not an official KITTI-360 release.** It contains nine `2013_05_28_drive_XXXX_sync` sequences with 1408×376 PNG images under `image_00/data` (KITTI-360 pixel dimensions, not KITTI-raw's 1242×375), Velodyne scans under `data_3d_raw`, and 3D bounding boxes. It has **no** official `calibration/`, `data_poses/`, `data_2d_raw/`, or `data_2d_semantics/`; instead it has KITTI-raw-style `calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`, and `calib_imu_to_velo.txt`. The `processed/` folders are empty scaffolding.
3. **Disk is the binding constraint.** `/data-20T` has 42 GB free; `/` has 55 GB free. The existing sequence `2013_05_28_drive_0000_sync` already occupies ~90 GB of images plus ~20 GB of LiDAR. Re-downloading official `data_2d_raw` for a full sequence is therefore not possible without deleting space. The plan must acquire the small official components (calibration, poses, semantics) and reuse the existing 1408×376 images **only after the G1 overlay proves they are geometrically correct rectified images**.
4. **Calibration truth is fully specified by the project research.** `poses.txt` is IMU→world, `calib_cam_to_pose.txt` is camera→IMU, `calib_cam_to_velo.txt` is camera-0→Velodyne, and `R_rect` comes from `perspective.txt`. `cam0_to_world.txt` is an assertion target, not a source of truth. This chain is normative for `docs/frames.md` and all code.
5. **The `< 2 px` overlay gate is a visual gate with an automated pre-check.** Projection math is unit-testable; the pixel-level calibration sign-off requires a human inspecting 4× crops of at least three frames. An automated depth-discontinuity ↔ image-edge proxy catches gross errors before the human check, but must not replace it.
6. **Container runtime is still TBD.** `AGENTS.md` forbids inventing the image/name/run command. Phase 1 must resolve the runtime contract before any data job or test executes.

---

## 1. Dataset Acquisition

### 1.1 Official layout (verified against the KITTI-360 documentation)

Source: <https://www.cvlibs.net/datasets/kitti-360/documentation.php> (fetched 2026-10-30).

| Component | Required for Phase 1 | Official paths |
|---|---|---|
| Calibration | **Yes (blocking)** | `calibration/calib_cam_to_pose.txt`, `calibration/calib_cam_to_velo.txt`, `calibration/perspective.txt` |
| Poses | **Yes (blocking)** | `data_poses/<seq>/poses.txt`, `data_poses/<seq>/cam0_to_world.txt` |
| Rectified perspective images | **Yes (blocking)** | `data_2d_raw/<seq>/image_00/data_rect/<frame:010d>.png`, `.../image_01/data_rect/...` |
| 2D semantics | **Yes (blocking)** | `data_2d_semantics/train/<seq>/image_00/semantic/<frame:010d>.png`, `.../image_01/semantic/...` |
| Raw LiDAR | **Yes** | `data_3d_raw/<seq>/velodyne_points/data/<frame:010d>.bin` (4×float32 per point) |
| Fisheye images | No | `image_02`/`image_03` — deferred to a later milestone |
| 3D semantics / bboxes | No | Used only if a later phase needs label transfer |

Frame IDs are 10-digit zero-padded integers. `poses.txt` is sparse: a pose exists only when the platform moved more than a threshold, so frame index ≠ line number. `cam0_to_world.txt` has its own, possibly different, frame list and is rectified-camera-0 → world.

### 1.2 Acquisition procedure

- Download portal: `https://www.cvlibs.net/datasets/kitti-360/user_login.php`
- Registration: `https://www.cvlibs.net/datasets/kitti-360/user_register.php`
- Terms: KITTI-360 derived assets are **non-commercial research use only**; the licence statement must be recorded in the readiness manifest.
- The portal exposes the dataset as archives. Archive granularity and byte sizes are not documented on the public pages; the plan records the actual archive names, sizes, and SHA-256 values at acquisition time rather than inventing them.
- **Do not** fetch KITTI-raw calibration (`calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`, `calib_imu_to_velo.txt`) and call it KITTI-360 calibration. This is the exact substitution the roadmap forbids.

### 1.3 Local tree audit (verified 2026-10-30)

| Observation | Value | Consequence |
|---|---|---|
| Sequences present | 9 (`0000`, `0002`–`0007`, `0009`, `0010`) | Reuse candidate exists |
| Perspective image size | 1408×376 PNG | Matches KITTI-360 rectified dimensions |
| Image location | `raw/2013_05_28/<seq>/image_00/data/` and `image_01/data/` | Not the official `data_rect` / `data_rgb` names |
| Velodyne | `/data-20T/kitti/data_3d_raw/<seq>/velodyne_points/data/*.bin` | Available and can be linked |
| Official calibration | Absent | Must acquire |
| Official poses | Absent (`poses.txt`, `cam0_to_world.txt` not found) | Must acquire |
| 2D semantics | Absent | Must acquire |
| KITTI-raw calibration | Present at `raw/2013_05_28/` | Must be rejected, never used |
| `processed/*` | Empty directories | Not usable output |
| Disk free | 42 GB on `/data-20T`, 55 GB on `/` | Cannot re-download ~90 GB of images without cleanup |
| Sequence `0000` footprint | ~90 GB images, ~20 GB LiDAR | Reuse candidate |
| Container runtime | Not defined | Resolution required before execution |

### 1.4 Acquisition policy (recommended, subject to the Phase 1 checkpoint)

1. Canonical data root: `/data-20T/hybrid-lidar-sim/raw/kitti360/` (per `AGENTS.md` and README §6.2).
2. Acquire the **small official components**: `calibration/`, `data_poses/<seq>/`, `data_2d_semantics/train/<seq>/`.
3. Reuse existing images by creating **named symlinks** into the canonical root:
   - `data_2d_raw/<seq>/image_00/data_rect` → `/data-20T/kitti/raw/2013_05_28/<seq>/image_00/data`
   - `data_2d_raw/<seq>/image_01/data_rect` → `/data-20T/kitti/raw/2013_05_28/<seq>/image_01/data`
   - `data_3d_raw/<seq>/velodyne_points` → `/data-20T/kitti/data_3d_raw/<seq>/velodyne_points`
4. Declare image provenance as `local_repackaged_1408x376`, and **promote it to acceptable only if G1 overlays pass < 2 px**. The readiness manifest must carry this provenance; it must never claim byte-identity with the official release.
5. If the user rejects reuse, the only honest alternative is to free at least ~100 GB and download official `data_2d_raw` for the chosen sequence. Do not silently substitute KITTI-raw calibration or OXTS-derived poses.
6. Install **one sequence only**. Archives that contain other sequences are filtered by an allowlist during extraction; other sequence trees are not created.

Candidate sequence for calibration: `2013_05_28_drive_0000_sync` (already local). Final low-dynamic segment selection remains a Phase 2 requirement (DATA-03); Phase 1 records the sequence as *provisional for calibration*.

---

## 2. Normative Coordinate Chain

Notation: `T_AB` maps points from frame **B** to frame **A**, `p_A = T_AB @ p_B`. All matrices are 4×4 float64. Code must use explicit destination/source names, never bare `T_CL`.

### 2.1 Parsing rules

| File | Format |
|---|---|
| `calib_cam_to_pose.txt` | Lines `image_00:` … 12 numbers (3×4); append `[0,0,0,1]` |
| `calib_cam_to_velo.txt` | 12 numbers (3×4); append `[0,0,0,1]` |
| `perspective.txt` | `P_rect_00:` / `P_rect_01:` 12 numbers (3×4); `R_rect_00:` / `R_rect_01:` 9 numbers (3×3) |
| `poses.txt` | First integer = frame, next 12 numbers = 3×4 `T_W_I` |
| `cam0_to_world.txt` | First integer = frame, next 16 numbers = 4×4 `T_W_C0_rect` |

### 2.2 Composition (single source of truth)

```python
T_V_C0 = load_calib_cam_to_velo()          # p_V = T_V_C0 @ p_C0_raw
T_I_Ck = load_calib_cam_to_pose()          # p_I = T_I_Ck[k] @ p_Ck_raw
T_W_I  = poses[frame]                      # p_W = T_W_I @ p_I
R_rect4(k) = [[R_rect_k, 0], [0, 1]]

T_C0_Ck     = inv(T_I_C0) @ T_I_Ck[k]
T_V_Ck      = T_V_C0 @ T_C0_Ck
T_Ck_V      = inv(T_V_Ck)
T_W_V       = T_W_I @ T_I_C0 @ inv(T_V_C0)
T_W_Ck_rect = T_W_I @ T_I_Ck[k] @ inv(R_rect4(k))
T_Ck_rect_V = inv(T_W_Ck_rect) @ T_W_V      # == R_rect4(k) @ T_Ck_V
T_S_W       = translation(-median_camera_center_W)
```

Assertions: `T_AB @ T_BA ≈ I` (< 1e-9), `det(R) = +1`, `RᵀR = I` (< 1e-6), and `T_W_C0_rect` from the chain vs `cam0_to_world.txt` (< 1e-3). `cam0_to_world.txt` is used **only** for this assertion.

### 2.3 Camera height

KITTI-360 world `z` is up, but the world origin is not the road surface. Camera height is therefore measured in the Velodyne frame against a coarse ground estimate:

```python
# Deterministic coarse ground estimate from one scan.
pts = load_velodyne(frame)                      # (N,4) float32, x fwd / y left / z up
xy = hypot(pts[:, 0], pts[:, 1])
candidate = pts[(xy > 5) & (xy < 30) & (pts[:, 2] > -3.0) & (pts[:, 2] < 1.0)]
ground_z_V = median(candidate[:, 2])
camera_center_V = T_V_C0[:3, 3]
height_m = camera_center_V[2] - ground_z_V
```

The G1 acceptance band is **1.4–1.9 m**, checked on ≥3 sampled frames and required to be stable (max − min ≤ 0.1 m across the sampled frames). This is a sanity check, not a survey-grade measurement; it catches sign errors, mirrored frames, and wrong extrinsic files.

### 2.4 Rectification and image_01

- `data_rect` images are rectified; `data_rgb` images are raw. Never mix `data_rgb` with `R_rect`/`P_rect`.
- Projection: `p_Crect = R_rect_k @ (T_Ck_V @ p_V)`, then `u = K[0,0]·x/z + K[0,2]`, `v = K[1,1]·y/z + K[1,2]`, with `K = P_rect_k[:3, :3]`.
- Do **not** use the full 3×4 `P_rect` for a single camera; its fourth column contains the stereo baseline.
- For `image_01`, use its own rectified pose `T_W_C01_rect` and `K = P_rect_01[:3,:3]`; do not add `Tx` twice.

---

## 3. Validation Design

### 3.1 Preflight contract (DATA-01)

`hls check-dataset --config configs/dataset.yaml` emits a JSON report and a human summary. Exit codes:

| Code | Meaning |
|---|---|
| 0 | Every required component present |
| 2 | Blocking component missing: calibration, poses, or semantics |
| 3 | Only non-blocking component missing: images or LiDAR |
| 4 | Unsafe/invalid input: malformed calibration, KITTI-raw calibration presented as KITTI-360, unreadable config |

The report must list every required path as present/missing, per-component file counts and bytes, and any rejected KITTI-raw calibration files. A KITTI-raw calibration directory must never produce exit 0.

### 3.2 G1 transform assertions (DATA-04)

Machine-checkable and unit-testable on synthetic fixtures:

- round-trip `< 1e-9` for every composed transform pair;
- `det(R) = +1` within `1e-12`;
- `RᵀR = I` within `1e-6`;
- `T_W_C0_rect` chain vs `cam0_to_world.txt` within `1e-3`;
- camera height 1.4–1.9 m and stable across sampled frames;
- pose lookup returns `(pose_frame, T_W_I, delta)`; missing frames select the nearest valid pose and log the delta; never fall back to frame 0.

### 3.3 Overlay validation (DATA-09)

Two layers:

1. **Automated pre-check (advisory):** project LiDAR points with range 10–30 m; detect depth discontinuities between neighbouring projected points (> 0.5 m); measure their distance to the nearest image-gradient pixel using a distance transform; report median and P90 in pixels. A gross failure (> 5 px) blocks before human inspection. This metric is noisy on textureless surfaces and is **not** the gate by itself.
2. **Human sign-off (the gate):** render 4× crops for at least 3 frames and both cameras with projected points coloured by range and a 1-px grid; the user measures the offset of ≥ 5 identifiable features per frame (pole edges, building corners, road markings) and records the maximum offset. The gate passes only if the maximum measured offset is `< 2 px` on every sampled frame/camera, and no raw/rectified mixing is present.

If the overlay fails because the local images are raw rather than rectified, the plan must fail loudly and return to acquisition (official `data_rect` or an explicit undistortion step). Never relax the threshold.

### 3.4 Pre-registration (EVAL-15)

`docs/decisions.md` is written **before** any reconstruction, with a UTC timestamp and a SHA-256 recorded in `/data-20T/hybrid-lidar-sim/artifacts/01/decisions.sha256`. It freezes:

- segment selection criteria (lowest dynamic fraction, contiguous multi-stream coverage, residential/suburban preference);
- frame-window policy (60–120 consecutive frames, ≥ 30 evaluated frames, ≥ 20 000 valid rays/frame);
- τ values (0.05 / 0.20 / 0.50 m true Euclidean, primary 0.20 m);
- buckets ([0,20), [20,40), [40,50] primary; (50,60] secondary);
- exclusions (dynamic/ego, r < 1.5 m, r > 60 m, behind-camera/out-of-FOV with 5 px margin, occluded variant);
- primary metric (micro-average range RMSE over hits, paired with ray recall/coverage; CD-L1 and F@0.20 secondary).

Any later change requires a new hash, a documented rationale, and a new run ID.

---

## 4. Architecture and Module Seams

```
src/hls/
  cli.py                 # subcommand discovery via hls.commands
  commands/              # one module per command, exposes register(subparsers)
  common/io.py           # hashes, JSON, YAML
  common/frames.py       # T_AB algebra, SEG, pose lookup, assertions
  common/calib.py        # KITTI-360 file loaders + transform composition
  common/prereg.py       # decisions hash + verification
  data/layout.py         # official/repackaged component inventory
  data/preflight.py      # exit contract + readiness report
  data/acquire.py        # safe extraction/installation + manifest
  data/overlay.py        # projection, overlay render, alignment proxy
configs/                 # dataset.yaml, frames.yaml, protocol.yaml
tests/                   # synthetic fixtures; no real data required for units
docs/frames.md           # normative transform reference
docs/decisions.md        # pre-registered protocol
scripts/                 # thin wrappers around `hls`
# Generated reports/overlays/manifests live under
# /data-20T/hybrid-lidar-sim/artifacts/01/ (AGENTS.md §3); the repo keeps none.
```

All run/job commands execute through `scripts/hls.sh`, which sources `.env.container` and refuses to run if `HLS_CONTAINER_RUN_CMD` is unset. Unit tests are also run through this wrapper to honour `AGENTS.md`.

---

## 5. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Download portal requires login and user has no account | Blocking | Explicit human-action checkpoint; registration URL and exact file list provided |
| User substitutes KITTI-raw calibration | Critical (silent metric error) | Preflight rejects the three KITTI-raw filenames; readiness manifest records provenance |
| Local images are raw, not rectified | High (5–20 px offset) | G1 overlay gate; fail loudly and acquire `data_rect` or add undistortion |
| Disk exhaustion during acquisition | Blocking | Acquire only small components; reuse validated local images; checkpoint before any large download |
| `poses.txt` sparse → wrong pose silently reused | Critical | Pose lookup logs `pose_frame`/`delta`; frames.csv records every choice; no frame-0 fallback |
| `P_rect` baseline added twice for image_01 | High | `K = P_rect[:3,:3]` enforced in code and tested |
| Container runtime unknown | Blocking | First plan is a decision checkpoint that records the exact run command |
| Threshold loosening to pass G1 | Critical | Thresholds live in config and are asserted in code; human sign-off records measured values |

---

## 6. Validation Architecture

### Test infrastructure

| Property | Value |
|---|---|
| Framework | pytest (Python 3.11) |
| Config | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run | `bash scripts/hls.sh python -m pytest tests -q` |
| Full suite | `bash scripts/hls.sh python -m pytest tests -q` |
| Fixtures | synthetic calibration/pose/image trees under `tests/fixtures/`; no real data required |
| Real-data gates | `/data-20T/hybrid-lidar-sim/artifacts/01/` reports + overlay PNGs + human sign-off |

### Requirement → validation map

| Requirement | Automated proof | Real-data proof | Manual proof |
|---|---|---|---|
| DATA-01 | unit tests for complete/missing/KITTI-raw/repackaged fixtures | `check-dataset` exit 2 on the current tree, exit 0 after acquisition | — |
| DATA-02 | extraction allowlist and manifest unit tests | readiness manifest with sizes + SHA-256; only one sequence installed | user confirms licence/download |
| DATA-04 | transform algebra tests (round-trip, det, orthonormality, pose lookup) | `calib_report.json` on the chosen sequence | — |
| DATA-09 | projection math tests + synthetic overlay score | overlay PNGs on ≥3 frames; automated proxy report | 4× visual measurement < 2 px |
| EVAL-15 | hash/verify tests; tamper changes hash | `decisions.sha256` + timestamp | — |

### Manual-only verification

| Behavior | Why manual | Instructions |
|---|---|---|
| Overlay alignment < 2 px | Pixel-level registration on real imagery is a visual measurement | Inspect `/data-20T/hybrid-lidar-sim/artifacts/01/overlays/frame_<frame:010d>/<cam>_overlay_4x.png`; measure ≥ 5 features per frame; record max offset |
| Dataset readiness sign-off | Confirms the correct official release was acquired under the licence | Review `/data-20T/hybrid-lidar-sim/artifacts/01/readiness.json` against the file list in §1.1 |

---

## Sources

- KITTI-360 documentation (layout, calibration semantics, poses, sensor axes): <https://www.cvlibs.net/datasets/kitti-360/documentation.php>
- KITTI-360 download login/registration: <https://www.cvlibs.net/datasets/kitti-360/user_login.php>
- `kitti360Scripts` calibration loader (`readVariable`, `loadCalibrationCameraToPose`, `loadPerspectiveIntrinsic`): <https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/devkits/commons/loadCalibration.py>
- Project research: `.planning/research/ARCHITECTURE.md` §Coordinate Frames & Calibration, `.planning/research/SUMMARY.md` §Phase 1, `.planning/research/PITFALLS.md` C1/C2/C5
- Local audit of `/data-20T/kitti` and disk state, 2026-10-30
