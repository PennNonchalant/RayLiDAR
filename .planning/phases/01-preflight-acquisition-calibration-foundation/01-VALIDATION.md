---
phase: 1
slug: preflight-acquisition-calibration-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-30
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x (Python 3.11) |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` (created in Plan 01-01, Wave 1) |
| **Quick run command** | `bash scripts/hls.sh python -m pytest tests -q` |
| **Full suite command** | `bash scripts/hls.sh python -m pytest tests -q` |
| **Estimated runtime** | ~20–40 seconds (synthetic fixtures only) |

All commands run through `scripts/hls.sh`, which exports `PYTHONPATH=$ROOT/src`, enforces the project-container execution contract from `AGENTS.md`, and sets `HLS_ARTIFACTS_ROOT=/data-20T/hybrid-lidar-sim/artifacts` (no generated artifacts in the repo).

---

## Sampling Rate

- **After every task commit:** Run `bash scripts/hls.sh python -m pytest tests -q`
- **After every plan wave:** Run `bash scripts/hls.sh python -m pytest tests -q`
- **Before `/gsd-verify-work`:** Full suite must be green plus the G1 real-data report
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01-01 | 1 | — | T-1-05 | Container runtime recorded; no invented image/command | manual | `test -s .env.container && grep -q '^HLS_CONTAINER_RUN_CMD=' .env.container` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01-01 | 1 | DATA-02 | T-1-02 | Acquisition policy records root/artifacts/links/provenance, no KITTI-raw substitution | manual | `grep -q '^artifacts_root:' /data-20T/hybrid-lidar-sim/artifacts/01/execution_contract.md && grep -q '^link_sources:' /data-20T/hybrid-lidar-sim/artifacts/01/execution_contract.md` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01-01 | 1 | DATA-01 | T-1-05 | Scaffold, PYTHONPATH wrapper, CLI discovery (`__main__` guard), configs, pytest harness; canonical dataset root exists; no secrets committed | unit | `bash scripts/hls.sh python -m pytest tests -q && test "$(bash scripts/hls.sh hls version)" = "0.1.0" && test -d /data-20T/hybrid-lidar-sim/raw/kitti360` | ❌ W0 | ⬜ pending |
| 1-02-01 | 01-02 | 2 | DATA-01 | T-1-02 | Rejects KITTI-raw calibration (incl. external scan root); allowlisted component inventory | unit | `bash scripts/hls.sh python -m pytest tests/common/test_io.py tests/data/test_layout.py tests/data/test_preflight.py -q` | ❌ W0 | ⬜ pending |
| 1-02-02 | 01-02 | 2 | DATA-01 | T-1-02 | Fail-loud exit 2 on missing blocking components; exit 3 for rejected non-blocking components | integration | `bash scripts/hls.sh hls check-dataset --config configs/dataset.yaml; test $? -eq 2` | ❌ W0 | ⬜ pending |
| 1-03-01 | 01-03 | 3 | DATA-04 | — | Transform algebra, pose lookup, orthonormality assertions | unit | `bash scripts/hls.sh python -m pytest tests/common/test_frames.py tests/common/test_calib.py -q` | ❌ W0 | ⬜ pending |
| 1-03-02 | 01-03 | 3 | DATA-04 | — | `calib.npz`/`frames.csv` schema and CLI | unit | `bash scripts/hls.sh python -m pytest tests/commands/test_check_calib.py -q` | ❌ W0 | ⬜ pending |
| 1-04-01 | 01-04 | 3 | EVAL-15 | T-1-04 | Protocol values frozen; tamper changes hash | unit | `bash scripts/hls.sh python -m pytest tests/common/test_prereg.py -q` | ❌ W0 | ⬜ pending |
| 1-04-02 | 01-04 | 3 | EVAL-15 | T-1-04 | Hash/timestamp verification CLI | integration | `bash scripts/hls.sh hls prereg verify --decisions docs/decisions.md --sidecar /data-20T/hybrid-lidar-sim/artifacts/01/decisions.sha256` | ❌ W0 | ⬜ pending |
| 1-05-01 | 01-05 | 3 | DATA-02 | T-1-05 | Per-component/per-sequence archives staged; capacity budget; link sources confirmed (null allowed, `REQUIRED_USER_CONFIRM` not) | manual | `bash scripts/hls.sh python -c "import pathlib,sys,yaml; p=pathlib.Path('/data-20T/hybrid-lidar-sim/staging'); d=yaml.safe_load(pathlib.Path('configs/dataset.yaml').read_text())['dataset']; ls=d['link_sources']; ok=all(v is None or (isinstance(v,str) and v not in ('REQUIRED_USER_CONFIRM','')) for v in ls.values()); sys.exit(0 if p.is_dir() and any(p.iterdir()) and ok else 1)"` | ❌ W0 | ⬜ pending |
| 1-05-02 | 01-05 | 3 | DATA-02 | T-1-01, T-1-02 | Safe extraction (no traversal/symlinks), allowlist-aware disk guard, mapping, manifest, `--link-source` CLI | unit | `bash scripts/hls.sh python -m pytest tests/data/test_acquire.py -q` | ❌ W0 | ⬜ pending |
| 1-05-03 | 01-05 | 3 | DATA-02 | T-1-02 | Readiness manifest complete; only one sequence installed; no `REQUIRED_USER_CONFIRM` | integration | `bash scripts/hls.sh hls check-dataset --config configs/dataset.yaml --strict && ! grep -q 'REQUIRED_USER_CONFIRM' configs/dataset.yaml` | ❌ W0 | ⬜ pending |
| 1-06-01 | 01-06 | 4 | DATA-09 | T-1-06 | Projection math + synthetic alignment proxy | unit | `bash scripts/hls.sh python -m pytest tests/data/test_overlay.py -q` | ❌ W0 | ⬜ pending |
| 1-06-02 | 01-06 | 4 | DATA-09 | T-1-06 | Overlay CLI emits JSON + PNG with bounded image reads | integration | `bash scripts/hls.sh python -m pytest tests/commands/test_overlay_check.py -q` | ❌ W0 | ⬜ pending |
| 1-07-01 | 01-07 | 5 | DATA-04, DATA-09 | T-1-23 | G1 aggregator: expected-pair coverage, duplicate rejection, rectified-only, pending-human state | unit | `bash scripts/hls.sh python -m pytest tests/common/test_g1.py tests/commands/test_g1_check.py -q` | ❌ W0 | ⬜ pending |
| 1-07-02 | 01-07 | 5 | DATA-04, DATA-09 | T-1-23 | Real-sequence transform assertions and exactly `frames × cams` rectified overlays | integration | `bash scripts/hls.sh hls check-calib --config configs/frames.yaml --frames <f0,f1,f2>` and `hls overlay-check` | ❌ W0 | ⬜ pending |
| 1-08-01 | 01-08 | 6 | DATA-09 | T-1-22 | Human measures ≥5 features for every expected (frame,camera) pair; rectified provenance | manual | `bash scripts/hls.sh python -c "import json; from hls.common.g1 import validate_overlay_verification, expected_pairs_from_overlay; o=json.load(open('/data-20T/hybrid-lidar-sim/artifacts/01/overlays/overlay_report.json')); d=json.load(open('/data-20T/hybrid-lidar-sim/artifacts/01/overlay_verification.json')); raise SystemExit(0 if not validate_overlay_verification(d, expected=expected_pairs_from_overlay(o)) else 1)"` | ❌ W0 | ⬜ pending |
| 1-08-02 | 01-08 | 6 | DATA-09, EVAL-15 | T-1-04, T-1-24 | G1 report ties hashes, thresholds, and evidence; sign-off | integration | `bash scripts/hls.sh hls g1-check --config configs/frames.yaml --verification /data-20T/hybrid-lidar-sim/artifacts/01/overlay_verification.json --out /data-20T/hybrid-lidar-sim/artifacts/01/G1_REPORT` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` + `tests/conftest.py` + `tests/test_cli.py` + `scripts/hls.sh` + `.env.container` — real pytest harness, CLI discovery, and container wrapper with `PYTHONPATH` (Plan 01-01)
- [ ] `tests/common/test_io.py` + `tests/data/conftest.py` fixture factory — shared IO and synthetic dataset trees (Plan 01-02)
- [ ] `tests/fixtures/` synthetic calibration/pose/image trees — no real data needed for unit tests (Plans 01-02/03/06)

*Wave 0 ownership: Plan 01-01 delivers the harness that makes `python -m pytest tests -q` runnable before Wave 2; Plan 01-02 adds only the per-domain test modules and fixture factory. `wave_0_complete` stays `false` until 01-01 actually executes; `nyquist_compliant` stays `false` until the harness has run green at least once.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Container runtime contract | — (AGENTS.md) | User must supply the approved image/run command; the agent must not invent it | Record `.env.container`; run `bash scripts/hls.sh python -V` and confirm the container starts with `/data-20T` mounted and `PYTHONPATH`/`HLS_ARTIFACTS_ROOT` forwarded |
| Official dataset download | DATA-02 | KITTI-360 requires account login and licence acceptance | Log in at the KITTI-360 portal, download **per-component/per-sequence** calibration + poses + semantics for the chosen sequence into `/data-20T/hybrid-lidar-sim/staging`, confirm archive names/sizes and free-space budget |
| Overlay alignment < 2 px | DATA-09 | Pixel-level registration on real imagery is a visual measurement | Inspect `/data-20T/hybrid-lidar-sim/artifacts/01/overlays/frame_<frame:010d>/<cam>_overlay_4x.png` for exactly 3 frames × 2 cameras; confirm every `image` path contains `data_rect`; measure ≥5 features per pair on the 4× PNG and **divide by `scale_factor` (4)** so `max_offset_px` is in original 1× pixels; set `offset_unit: "original_1x_px"` and record one entry per expected pair in `/data-20T/hybrid-lidar-sim/artifacts/01/overlay_verification.json`; sign off only if all < 2 px |
| Dataset readiness sign-off | DATA-02 | Confirms the correct official release and provenance | Review `/data-20T/hybrid-lidar-sim/artifacts/01/readiness.json` against the official file list; approve reuse provenance if local images passed G1 |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (harness owned by 01-01, fixtures by 01-02/03/06)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter **only after** the 01-01 harness has run green

**Approval:** pending
