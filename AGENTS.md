# AGENTS.md — Execution Rules for This Repository

This file is the operating contract for any agent (human-directed or subagent) working in
`hybrid-lidar-sim`. Read it before running anything. Project background lives in `README.md`
and `.planning/PROJECT.md`.

## 1. Upstream

This project is built on **RayLiDAR**: https://github.com/PennNonchalant/RayLiDAR

That organization repository is the canonical upstream and the default remote (`origin`) for
code provenance and ported components. https://github.com/Stabil1ze/RayLiDAR is a **fork** of
it (a contributor's working copy), not the upstream. Do not assume, cite, or pull in any other
upstream unless the user names it.

## 2. Run everything inside the project container

- All run/job tasks — preprocessing, training, reconstruction, ray casting, evaluation,
  benchmark jobs — MUST execute inside the project container.
- Do not run jobs directly on the host.
- **Container specifics are TBD.** The image, container name, volume mounts, and the exact
  invocation command have not been decided yet. Do not invent them. Ask the user, and record
  the answer in this file and in `PROJECT.md` once known.

## 3. Put all data and artifacts under `/data-20T`

- `/data-20T` is the data disk.
- All datasets, preprocessed scenes, checkpoints, run outputs, metrics, plots, and other
  generated artifacts MUST live under `/data-20T` (recommended root:
  `/data-20T/hybrid-lidar-sim/`, matching README §6.2).
- Do not write large data or outputs to the repository tree or the system disk. Code stays in
  the repo; data does not.

## 4. Delegate parallelizable work to parallel subagents

- Independent tasks (different files, no ordering dependency) MAY be delegated to parallel
  subagents to cut wall-clock time.
- Keep work sequential when tasks touch the same files, share mutable state, or have a
  dependency order. Do not parallelize for its own sake.

## 5. Never guess — ask

- If project information is unknown or ambiguous (paths, container details, dataset layout,
  hardware, time budget, upstream forks), STOP and ask the user before executing.
- Do not fabricate values to unblock yourself. A recorded "unknown" plus a question is always
  better than a plausible invention.

## Open items

- [ ] Container image / name / run command — TBD
- [ ] Confirm `/data-20T/hybrid-lidar-sim/` as the canonical data root
