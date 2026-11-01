# Simulation I/O Contract (Phase 3, `schema_version` 1)

This document is the single normative contract for the three simulation-facing files that
member 1 (Vulkan framework), member 2 (GPU ray query), and this project exchange:

- `gt_rays.bin` — the world-frame camera-origin GT ray buffer,
- `hits.bin` — the per-ray prediction buffer produced by a simulator,
- mesh files — binary PLY, GLB 2.0, and OBJ triangle meshes.

All three are expressed in the **W frame**, in **metres**, with no axis conversion, no scale,
and no embedded model transform. A producer and a consumer that agree on this document agree on
every byte. A companion `sim_io_manifest.json` records the `schema_version`, kind, path,
SHA-256, frame, units, and shape of each exported file.

If this document is edited, `docs/sim-io.sha256` MUST be regenerated; a stale sidecar is a test
failure (`tests/io/test_sim_io_doc.cpp`).

## 1. Frames and units

- **`W` frame.** `W` is the `poses.txt` world frame: metres, right-handed, **x forward, y left,
  z up**. No consumer may re-express, rotate, or rescale the data; there is no second frame in
  this contract.
- **Metres.** Every translation, range, and bounding-box value is in metres. Angles, if any,
  are radians; the simulation I/O contract itself carries no angles.
- **Per-ray origin.** Every `gt_rays.bin` record carries its own `origin_W` — the camera-centre
  origin for that ray. The two cameras and every frame have different centres, so a consumer
  **MUST NOT assume a single sensor origin**; it must read `origin_W` from each record.
- **Winding and sidedness.** Triangle winding is counter-clockwise as seen from outside when the
  mesh is closed, but casting is two-sided and a consumer **MUST NOT back-face cull** (MVS meshes
  are open and inconsistently oriented). Meshes are static: no animation, skinning, or scene
  graph semantics.

## 2. Common binary rules

- **Byte order.** All binary files are **little-endian**. Integers are fixed-width
  two's-complement. Distances and coordinates are IEEE-754 `float64` (binary64).
- **Header (32 bytes, both `gt_rays.bin` and `hits.bin`).**

  | Offset | Bytes | Field |
  |-------:|------:|-------|
  | 0  | 4  | `magic[4]` ASCII (`RLGR` or `RLHT`) |
  | 4  | 2  | `version` `u16` (1) |
  | 6  | 2  | `record_size` `u16` (96 or 24) |
  | 8  | 4  | `count` `u32` (number of records) |
  | 12 | 20 | `reserved[20]` (zero-filled) |

  Reserved bytes MUST be zero on write and are ignored on read. The exact file length MUST be
  `32 + count * record_size`; a shorter or longer file is invalid.
- **`schema_version`.** The manifest field `schema_version` is `1` for this contract.
- **Manifest.** The companion `sim_io_manifest.json` records `schema_version`, `kind`
  (`gt_rays` | `hits` | `mesh`), `path`, `sha256` (lowercase hex of the file bytes), `version`,
  `record_size`, `count`, `frame` (`W`), `units` (`m`), and, for a mesh, `format`,
  `vertex_count`, `triangle_count`, `bbox_min_m`, `bbox_max_m`.
- **Failure classes.** A hash mismatch detected at a CLI verify boundary is a
  **schema/hash failure** (exit 6). A structurally invalid or semantically invalid file is
  **invalid input** (exit 4, `raylidar::IoError`).

## 3. `gt_rays.bin`

- Magic `RLGR`, `version` 1, `record_size` 96, header 32 bytes.
- One record per GT ray:

  | Offset | Bytes | Field |
  |-------:|------:|-------|
  | 0  | 8  | `ray_id` `u64` (unique) |
  | 8  | 24 | `origin_W` `float64[3]` (camera centre for this ray) |
  | 32 | 24 | `dir_W` `float64[3]` (unit direction) |
  | 56 | 8  | `t_gt` `float64` (true range from `origin_W`) |
  | 64 | 8  | `t_min` `float64` (self-hit guard, frozen `1e-4` m) |
  | 72 | 8  | `t_max` `float64` (cast bound, frozen `60` m) |
  | 80 | 4  | `frame` `u32` |
  | 84 | 2  | `camera` `u16` (0 = `image_00`, 1 = `image_01`) |
  | 86 | 10 | `reserved[10]` (zero-filled) |

- Semantics and validation:
  - `origin_W`, `dir_W`, `t_gt`, `t_min`, and `t_max` MUST be finite.
  - `dir_W` MUST be unit within `1e-6` (`| ‖dir_W‖ − 1 | ≤ 1e-6`).
  - `0 < t_min` and `t_min ≤ t_gt ≤ t_max`.
  - `ray_id` MUST be unique; `camera` MUST be 0 or 1.
  - `frame` accepts any `u32` (frame 0 is valid for synthetic fixtures).
- **Provenance.** The frozen Phase-2 ray set is the provenance of these rays, but the W-frame
  `gt_rays.bin` export itself is Phase 5 (IO-02); this document freezes the layout only.

## 4. `hits.bin`

- Magic `RLHT`, `version` 1, `record_size` 24, header 32 bytes.
- One record per GT ray, exactly one hit record per `gt_rays.bin` record, in the same order and
  with equal `ray_id`:

  | Offset | Bytes | Field |
  |-------:|------:|-------|
  | 0  | 8  | `ray_id` `u64` |
  | 8  | 8  | `t_pred` `float64` |
  | 16 | 1  | `hit` `u8` (0 or 1) |
  | 17 | 3  | `reserved[3]` (zero-filled) |
  | 20 | 4  | `prim_id` `u32` |

- **Hit record.** `hit = 1`, `t_pred` finite and inside `[t_min, t_max]`, `prim_id` is the
  0-based triangle id, or `0xFFFFFFFF` when the producer does not expose one.
- **Miss record.** `hit = 0`, `t_pred = +inf`, `prim_id = 0xFFFFFFFF`.
- **No other miss convention.** A miss is never `0`, never NaN, and never a clipped number.
  Misses never enter a range mean.
- The metrics attribution of `hits.bin` is settled in Phase 6; Phase 3 freezes the format only.

## 5. Mesh formats

All three formats carry vertex positions in the **W frame** in **metres**, with no node or
object transform. A mesh MUST have at least one vertex and one triangle, and every coordinate
MUST be finite.

### PLY

- `format binary_little_endian 1.0` is required. ASCII PLY and big-endian PLY are invalid.
- Required comments: `comment frame W` and `comment units m`.
- Vertex properties `x`, `y`, `z` as `float` or `double` (`float64` positions are recorded;
  `float32` positions are accepted). Faces are triangle index lists with exactly three 0-based
  in-range indices.
- Bounding box is computed from all vertex positions.

### GLB

- glTF 2.0 container: `glTF` magic, container version 2, JSON + BIN chunks, total length equal
  to the file size.
- `asset.version` is `"2.0"`; `asset.extras.frame` is `"W"`; `asset.extras.units` is `"m"`.
- Every node transform is identity: no `matrix`, `translation`, `rotation`, or `scale`.
- Each `POSITION` accessor is component type `5126` (`float32`) with required finite `min` and
  `max` arrays. Triangles are indexed (`count / 3`) or non-indexed (`POSITION.count / 3`).
- Morph targets, skins, animations, and sparse accessors are invalid.

### OBJ

- ASCII. Required first-line comments before any vertex: `# frame W` and `# units m`.
- `v` positions are in metres. Faces are triangles only: exactly three positive 1-based indices
  per `f`. Negative indices, zero indices, out-of-range indices, and polygon (quad) faces are
  invalid. `mtllib` is optional and may be ignored by a loader; geometry must not depend on it.
- Transform directives (`o`, `g`, `vp`, `l`, `p`) are invalid because they are not geometry in
  the frozen contract. `vn`/`vt` are allowed and ignored (normals and texture coordinates are
  not part of the frozen geometry).

## 6. Validation and exit codes

A reader MUST validate the exact file length `32 + count * record_size` **before** allocating
record storage, and MUST reject a `count` that would overflow that expression or exceed the
implementation cap (`kMaxRecordCount = 200000000`). Checked `uint64` arithmetic is required.

Reject conditions (all map to `raylidar::IoError` with exit code 4):

- `gt_rays.bin` / `hits.bin`: wrong magic, wrong version, wrong record size, a count/length
  mismatch (short or long file), a count above the cap, a non-finite value, a non-unit
  direction, `t_min ≤ 0`, `t_gt` outside `[t_min, t_max]`, an unknown camera, a duplicate or
  reordered `ray_id`, an invalid `hit` value, a hit with a non-finite or out-of-bound `t_pred`,
  or a miss without `+inf` and the `0xFFFFFFFF` sentinel.
- Meshes: ASCII or big-endian PLY, missing `frame`/`units` metadata, a non-finite coordinate,
  zero vertices or triangles, an out-of-range face index, a non-triangle OBJ face, a negative
  OBJ index, a GLB node transform, a sparse accessor, or a missing/float64 `POSITION`.

Schema/hash mismatches at a CLI verify boundary map to exit code 6. Path handling takes explicit
CLI/test paths only; no archive extraction, dynamic `mtllib` fetch, or shell execution occurs in
this phase.

## Out of scope

The Phase-3 deliverable is this document plus the shared encoders, decoders, and validators.
Explicitly excluded:

- real mesh loading and any mesh export (IO-03, Phase 4),
- `gt_rays.bin` export from the frozen ray set (IO-02, Phase 5),
- a GPU shader or a GPU Ray Query (`VK_KHR_ray_query`) implementation,
- plots, the Markdown report, and any other reports (Phase 6).
