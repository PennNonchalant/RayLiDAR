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
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` (created in Wave 1) |
| **Quick run command** | `bash scripts/hls.sh pytest tests -q` |
| **Full suite command** | `bash scripts/hls.sh pytest tests -q` |
| **Estimated runtime** | ~20–40 seconds (synthetic fixtures only) |

All commands run through `scripts/hls.sh`, which enforces the project-container execution contract from `AGENTS.md`.

---

## Sampling Rate

- **After every task commit:** Run `bash scripts/hls.sh pytest tests -q`
- **After every plan wave:** Run `bash scripts/hls.sh pytest tests -q`
- **Before `/gsd-verify-work`:** Full suite must be green plus the G1 real-data report
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01-01 | 1 | — | T-1-05 | Container runtime recorded; no invented image/command | manual | `test -s .env.container` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01-01 | 1 | DATA-02 | T-1-02 | Acquisition policy records reuse/provenance, no KITTI-raw substitution | manual | `grep -q 'image_provenance' configs/dataset.yaml` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01-01 | 1 | DATA-01 | T-1-05 | Scaffold, CLI discovery, configs, wrapper; no secrets committed | unit | `bash scripts/hls.sh pytest tests/test_cli.py tests/common/test_io.py -q` | ❌ W0 | ⬜ pending |
| 1-02-01 | 01-02 | 2 | DATA-01 | T-1-02 | Rejects KITTI-raw calibration; allowlisted component inventory | unit | `bash scripts/hls.sh pytest tests/data/test_layout.py tests/data/test_preflight.py -q` | ❌ W0 | ⬜ pending |
| 1-02-02 | 01-02 | 2 | DATA-01 | T-1-02 | Fail-loud exit code 2 on missing official calibration/poses/semantics | integration | `bash scripts/hls.sh hls check-dataset --config configs/dataset.yaml; test $? -eq 2` | ❌ W0 | ⬜ pending |
| 1-03-01 | 01-03 | 3 | DATA-04 | — | Transform algebra, pose lookup, orthonormality assertions | unit | `bash scripts/hls.sh pytest tests/common/test_frames.py tests/common/test_calib.py -q` | ❌ W0 | ⬜ pending |
| 1-03-02 | 01-03 | 3 | DATA-04 | — | `calib.npz`/`frames.csv` schema and CLI | unit | `bash scripts/hls.sh pytest tests/commands/test_check_calib.py -q` | ❌ W0 | ⬜ pending |
| 1-04-01 | 01-04 | 3 | EVAL-15 | T-1-04 | Protocol values frozen; tamper changes hash | unit | `bash scripts/hls.sh pytest tests/common/test_prereg.py -q` | ❌ W0 | ⬜ pending |
| 1-04-02 | 01-04 | 3 | EVAL-15 | T-1-04 | Hash/timestamp verification CLI | integration | `bash scripts/hls.sh hls prereg verify --decisions docs/decisions.md --hash artifacts/01/decisions.sha256` | ❌ W0 | ⬜ pending |
| 1-05-01 | 01-05 | 3 | DATA-02 | T-1-05 | Official archives present in staging; licence accepted | manual | `ls -lh /data-20T/hybrid-lidar-sim/staging` | ❌ W0 | ⬜ pending |
| 1-05-02 | 01-05 | 3 | DATA-02 | T-1-01, T-1-02 | Safe extraction (no traversal/symlinks), allowlist, manifest | unit | `bash scripts/hls.sh pytest tests/data/test_acquire.py -q` | ❌ W0 | ⬜ pending |
| 1-05-03 | 01-05 | 3 | DATA-02 | T-1-02 | Readiness manifest complete; only one sequence installed | integration | `bash scripts/hls.sh hls check-dataset --config configs/dataset.yaml` | ❌ W0 | ⬜ pending |
| 1-06-01 | 01-06 | 4 | DATA-09 | T-1-06 | Projection math + synthetic alignment proxy | unit | `bash scripts/hls.sh pytest tests/data/test_overlay.py -q` | ❌ W0 | ⬜ pending |
| 1-06-02 | 01-06 | 4 | DATA-09 | T-1-06 | Overlay CLI emits JSON + PNG with bounded image reads | integration | `bash scripts/hls.sh pytest tests/commands/test_overlay_check.py -q` | ❌ W0 | ⬜ pending |
| 1-07-01 | 01-07 | 5 | DATA-04, DATA-09 | T-1-23 | G1 aggregator unit tests; pending-human state | unit | `bash scripts/hls.sh pytest tests/common/test_g1.py tests/commands/test_g1_check.py -q` | ❌ W0 | ⬜ pending |
| 1-07-02 | 01-07 | 5 | DATA-04, DATA-09 | T-1-23 | Real-sequence transform assertions and ≥3 frames/camera overlays | integration | `bash scripts/hls.sh hls check-calib --config configs/frames.yaml --frames <f0,f1,f2>` and `hls overlay-check` | ❌ W0 | ⬜ pending |
| 1-08-01 | 01-08 | 6 | DATA-09 | T-1-22 | Human measures ≥5 features per frame/camera | manual | review `artifacts/01/overlay_verification.json` schema (features ≥5, max_offset_px <2) | ❌ W0 | ⬜ pending |
| 1-08-02 | 01-08 | 6 | DATA-09, EVAL-15 | T-1-04, T-1-24 | G1 report ties hashes, thresholds, and evidence; sign-off | integration | `bash scripts/hls.sh hls g1-check --config configs/frames.yaml --verification artifacts/01/overlay_verification.json --out artifacts/01/G1_REPORT` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` + `tests/conftest.py` + `tests/test_cli.py` — pytest harness and CLI discovery (Plan 01-01)
- [ ] `tests/fixtures/` synthetic calibration/pose/image trees — no real data needed for unit tests (Plans 01-02/03/06)
- [ ] `scripts/hls.sh` + `.env.container` — container execution contract (Plan 01-01)

*Wave 0 is delivered by Plan 01-01 before any other plan's tests run.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Container runtime contract | — (AGENTS.md) | User must supply the approved image/run command; the agent must not invent it | Record `.env.container`; run `bash scripts/hls.sh python -V` and confirm the container starts with `/data-20T` mounted |
| Official dataset download | DATA-02 | KITTI-360 requires account login and licence acceptance | Log in at the KITTI-360 portal, download calibration + poses + semantics for the chosen sequence into `/data-20T/hybrid-lidar-sim/staging`, confirm archive names and sizes |
| Overlay alignment < 2 px | DATA-09 | Pixel-level registration on real imagery is a visual measurement | Inspect `artifacts/01/overlays/*/overlay_4x.png` for ≥3 frames; measure ≥5 features per frame; record max offset in `artifacts/01/overlay_verification.json`; sign off only if all < 2 px |
| Dataset readiness sign-off | DATA-02 | Confirms the correct official release and provenance | Review `artifacts/01/readiness.json` against the official file list; approve reuse provenance if local images passed G1 |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
