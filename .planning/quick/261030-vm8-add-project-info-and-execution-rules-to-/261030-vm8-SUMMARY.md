---
phase: quick
plan: 261030-vm8
subsystem: docs
tags: [documentation, operations, execution-rules, agents-md]
dependency_graph:
  requires: []
  provides:
    - "agent execution contract at repo root"
    - "operations rules mirrored in PROJECT.md and README.md"
  affects:
    - .planning/PROJECT.md
    - README.md
    - AGENTS.md
tech_stack:
  added: []
  patterns:
    - "AGENTS.md as the single imperative agent contract; PROJECT.md/README mirror it operationally"
key_files:
  created:
    - AGENTS.md
  modified:
    - .planning/PROJECT.md
    - README.md
decisions:
  - "Five operations rules recorded in all three project-facing docs, with AGENTS.md as the operative contract"
  - "Container image/name/run command deliberately left TBD rather than invented"
metrics:
  duration: "~8 minutes"
  completed: "2026-10-30"
  tasks_completed: 3
  files_changed: 3
requirements: []
---

# Phase quick Plan 261030-vm8: Project Info & Execution Rules Summary

Recorded upstream provenance and five execution/operations rules across `.planning/PROJECT.md`,
`README.md`, and a new root `AGENTS.md`, leaving all container specifics explicitly TBD.

## What Was Built

Three project-facing documents now carry the same five rules, at three levels of authority:

| File | Role | Change |
|------|------|--------|
| `AGENTS.md` | Operative agent contract (imperative, numbered) | Created |
| `.planning/PROJECT.md` | Planning source of truth | `### Operations & Execution Rules` under Context, `/data-20T` constraint bullet, one Key Decisions row |
| `README.md` | Human/agent entry point | New `## Project Operations & Execution Rules` section before `## 0. TL;DR`, linking to `AGENTS.md` |

## The Five Rules (as written)

1. **Upstream** — built on `https://github.com/Stabil1ze/RayLiDAR`; no other upstream assumed.
2. **Container-only execution** — all run/job tasks execute inside the project container; image,
   name, volume mounts, and invocation command are **TBD** and must not be invented.
3. **Data disk `/data-20T`** — all datasets, preprocessed artifacts, checkpoints, run outputs,
   metrics, and plots live under `/data-20T` (recommended root `/data-20T/hybrid-lidar-sim/`,
   consistent with README §6.2). Nothing large on the system disk.
4. **Parallel subagent delegation** — independent, file-disjoint tasks may be parallelized;
   file-conflicting or ordered work stays sequential.
5. **Never guess — ask** — unknown paths, container details, dataset layout, hardware, or
   upstream forks must be raised with the user rather than fabricated.

## Open Items Left for the User

- **Container image / name / run command — TBD.** Recorded as unresolved in both `AGENTS.md`
  (`## Open items`) and `.planning/PROJECT.md`. Deliberately not invented per AGENTS.md §5.
- **Confirm `/data-20T/hybrid-lidar-sim/` as the canonical data root** — proposed, not confirmed.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add operations & execution rules to PROJECT.md | `f359a52` | `.planning/PROJECT.md` |
| 2 | Add "Project Operations & Execution Rules" to README.md | `db69fa6` | `README.md` |
| 3 | Create root AGENTS.md with imperative agent instructions | `5379ea6` | `AGENTS.md` |

## Verification

All plan assertions passed:

- Upstream URL `https://github.com/Stabil1ze/RayLiDAR` present in all three files; no other upstream named.
- `### Operations & Execution Rules` in `PROJECT.md`; `## Project Operations & Execution Rules` in `README.md`; `AGENTS.md` present at repo root and linked from `README.md`.
- `/data-20T` stated as the data/artifact location in all three files.
- Container specifics marked TBD only — `grep` for `docker run|docker image|docker pull|ghcr.io|nvcr.io` found nothing.
- No structural regressions: `## 0. TL;DR` (README), `## Context` / `## Constraints` / `## Key Decisions` / `## Evolution` (PROJECT.md) and the `Last updated` footer all intact. Edits were insertions only.
- README §0–§12 untouched; §6.2 `/data-20T/hybrid-lidar-sim/` path unchanged.

## Deviations from Plan

None - plan executed exactly as written. The plan's `PROJECT.md` path resolved to
`.planning/PROJECT.md` (the only PROJECT.md in the repo); no root-level PROJECT.md was created.

## Known Stubs

None. No code, data source, or UI surface was introduced — this plan changed documentation only.

## Threat Flags

None. No new network, auth, file-access, or schema surface. Docs contain only a public upstream
URL and a local disk path, consistent with the plan's threat model (T-quick-01 accept,
T-quick-02 mitigations verified by the structural no-regression checks).

## TDD Gate Compliance

Not applicable — plan `type: execute`, documentation only, no tests.

## Self-Check: PASSED

- `AGENTS.md` — FOUND
- `README.md` — FOUND (tracked, committed in `db69fa6`)
- `.planning/PROJECT.md` — FOUND (modified, committed in `f359a52`)
- Commits `f359a52`, `db69fa6`, `5379ea6` — FOUND in `git log`
- No file deletions in the three task commits
