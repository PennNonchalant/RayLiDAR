# Normative Frame and Calibration Reference

This document is the single source of truth for the KITTI-360 transform algebra used by
Phase 1. Every module must use these conventions; no second convention is permitted.

Implemented in `raylidar/calib/frames.h` and `raylidar/calib/kitti360.h`.

## Notation

`T_AB` always means `p_A = T_AB * p_B`. That is, the **destination** frame comes first and the
**source** frame second. All transforms are `Eigen::Matrix4d` in float64 with a bottom row of
`[0, 0, 0, 1]`.

## Frames

| Frame | Meaning | Source |
|-------|---------|--------|
| `V` | Velodyne / LiDAR frame | `calib_cam_to_velo.txt` |
| `Ck_raw` | Raw (unrectified) camera `k` frame | `calib_cam_to_pose.txt` |
| `Ck_rect` | Rectified camera `k` frame | `perspective.txt` `R_rect_kk` |
| `I` | IMU / poses reference frame | `data_poses/<seq>/poses.txt` |
| `W` | World frame of `poses.txt` and `cam0_to_world.txt` | `poses.txt` |
| `S` | Scene-local (SEG) frame with origin at the median camera centre | computed |
| `Ek` | Per-camera virtual frame of the rectified pinhole | `perspective.txt` `P_rect_kk` |

## Parsing rules

- `calib_cam_to_pose.txt`: one `image_XX: r11 r12 r13 tx r21 ... tz` line per camera
  (12 floats), appended with a `[0, 0, 0, 1]` row.
- `calib_cam_to_velo.txt`: exactly 12 floats on non-comment lines, appended with `[0, 0, 0, 1]`.
- `perspective.txt`: `R_rect_XX` (9 floats) and `P_rect_XX` (12 floats). `K_rect` is always
  `P_rect.leftCols<3>()`; the full 3x4 `P_rect` is never used as a single-camera intrinsic.
- `poses.txt`: `frame r11 r12 r13 tx r21 ... tz` (13 tokens).
- `cam0_to_world.txt`: `frame` plus 16 floats (row-major 4x4).
- All loaders reject wrong token counts, duplicate frames, and non-finite values with a
  message naming the file and line. No loader defaults to frame 0.

## Composition

```
T_V_C0     = calib_cam_to_velo()                  # p_V     = T_V_C0     * p_C0_raw
T_I_Ck     = calib_cam_to_pose()[k]               # p_I     = T_I_Ck     * p_Ck_raw
T_W_I      = poses[frame]                         # p_W     = T_W_I      * p_I

T_C0_Ck    = inv(T_I_C0)  * T_I_Ck
T_V_Ck     = T_V_C0       * T_C0_Ck
T_Ck_V     = inv(T_V_Ck)
T_W_V      = T_W_I        * T_I_C0 * inv(T_V_C0)
T_W_Ck_rect= T_W_I        * T_I_Ck * inv(R_rect4(k))
T_Ck_rect_V= inv(T_W_Ck_rect) * T_W_V             # must equal R_rect4(k) * T_Ck_V
```

`ComposeTCkRectV` evaluates both the inverted-chain form and `R_rect4(k) * inv(T_V_Ck)` and
asserts they agree within `1e-9`. The two paths are independent of each other, so this catches
a composition or lookup mistake immediately.

## Rectification

`R_rect4(k)` is the 4x4 homogeneous lift of `R_rect_kk`. Rectified images in `data_rect/` are
already rectified, so overlays and projections must use `Ck_rect` and `K_rect`, never
`Ck_raw`. Mixing raw and rectified imagery is forbidden.

## Pose lookup

`poses.txt` is sparse. `LookupPose` uses an exact frame first, otherwise the nearest pose frame
(ties go to the lower frame), and throws `std::out_of_range` when the nearest delta exceeds
`max_delta_frames`. It never falls back to the first pose or frame 0. `check-calib` records the
requested frame, the selected pose frame, and the delta for every row so the choice is auditable.

## Common-frame window

`FindCommonFrames` returns one **contiguous local window**: frames that exist in `poses.txt`
and `cam0_to_world.txt` and have every camera's `data_rect` image and the Velodyne scan. The
first run of consecutive frame ids long enough wins; if no run is long enough, the longest run
is returned. It never samples first/middle/last across the sequence, because camera height
drifts by more than 0.1 m between the sequence start and middle.

## Camera height

Camera height is the perpendicular distance from the camera centre (the translation of
`T_V_C0`) to a robust local ground plane, not a raw `z` difference. The plane is estimated from
the LiDAR scan in `V`:

1. keep returns with horizontal radius in `[5, 30] m`;
2. bin by azimuth into 240 sectors and keep the single lowest-`z` return per sector (removes
   walls, cars, and tree canopies);
3. drop candidates outside `z in [-3, 1] m`;
4. fit `z = a*x + b*y + c` with 6 rounds of iteratively trimmed least squares (`trim_fraction
   = 0.7`);
5. normalise to `normal = normalize(-a, -b, 1)` and `offset = -c / ||(a, b, 1)||`;
6. require at least 100 candidates.

`CameraHeight` then returns `|normal . cameraCentreV + offset|`. A sloped local plane would
bias a raw z difference; the perpendicular distance does not.

## Measured source precision and tolerance choices

The official files are printed with limited precision:

| Quantity | Measured |
|----------|----------|
| `det(R_rect) - 1` | `4.52e-7` |
| `max |R_rect^T R_rect - I|` | `9.93e-7` |
| pose `det(R) - 1` | `<= 1.6e-10` |
| pose `max |R^T R - I|` | `<= 1.4e-10` |
| camera-relative round trip (`T_Ck_V`) | `~2.5e-10` |
| rectified/raw path agreement (`T_Ck_rect_V`) | `~1e-10` |
| `T_W_C0_rect` vs `cam0_to_world` | `<= 4.8e-4` |

Consequences, wired into `configs/frames.yaml`:

- The rotation/determinant gate is `2e-6` (`rotation_atol`, `rotation_det_atol`). A `1e-12`
  determinant gate is mathematically impossible with these files.
- The `1e-9` round-trip gate (`roundtrip_atol`) applies to **camera-relative** transforms and
  the rectified/raw path agreement. A literal `inv(T_Ck_rect_V) * T_Ck_rect_V` inherits the
  `~1e-6` `R_rect` orthonormality defect, so the rectified transform is validated by the two
  independent composition paths plus `AssertRotation`, not by transposing `R_rect`.
- World-frame transforms are checked rotation-only; their translations carry the same `~1e-6`
  source noise. `T_W_C0_rect` is cross-checked against `cam0_to_world.txt` at `1e-3`.
- `InvertRigid` compares its analytic inverse with the full 4x4 inverse using
  `atol + relAtol * max(1, ||t||)`, so the `~4 km` world translation is not compared against a
  fixed `1e-12`.

`cam0_to_world.txt` is assertion-only. It is never used to build the live transform chain; the
chain is always composed from `calib_cam_to_pose.txt`, `calib_cam_to_velo.txt`, `perspective.txt`,
and `poses.txt`.

## Artifacts

`raylidar_cli check-calib` writes `calib.json`, `frames.csv`, and `calib_report.json` under
`/data-20T/hybrid-lidar-sim/artifacts/01/calib/`. `frames.csv` writes every 3x4 matrix as 12
explicit components in `poses.txt` order (`r11, r12, r13, tx, r21, ... tz`), never as an
ambiguous 16- or 24-column block. `calib_report.json` records the measured values, thresholds,
per-check booleans, `max_cam0_world_delta`, the per-frame camera heights,
`camera_height_status`, `camera_height_spread_m`, and the overall `pass` flag.
