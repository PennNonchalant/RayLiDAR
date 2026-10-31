---
name: pr-workflow
description: Use when changing the RayLiDAR repository (PennNonchalant/RayLiDAR) — opening an issue, creating a branch, committing, pushing, or opening/merging a pull request. Enforces the issue → branch → PR → review flow.
---

# RayLiDAR PR workflow

`main` is protected. Every change follows: **issue → branch → PR → review → merge**.
Never push directly to `main`, and never merge a PR without an approving review.

Read `CONTRIBUTING.md` for the full policy; this skill is the executable checklist.

## Step 0 — Preflight

```bash
gh auth status                     # must be logged in
gh api repos/PennNonchalant/RayLiDAR --jq '.permissions.push'   # true = push directly; false = use a fork
git status --porcelain             # must be clean (or stash deliberately)
git fetch origin
```

Pick the push target from the permission result:

- `push: true` → push branches to `origin`.
- `push: false` → push to the fork and open the PR from `<you>:<branch>`.

## Step 1 — Issue first

```bash
gh issue create --repo PennNonchalant/RayLiDAR --assignee @me \
  --title "<area>: <description>" \
  --body "## Background
...
## Scope
...
## Acceptance criteria
- [ ] ..."
```

Record the issue number (`N`). Do not start editing before the issue exists.

## Step 2 — Branch from fresh main

```bash
git switch -c "N-<slug>" origin/main
```

Branch naming is mandatory: `<issue-number>-<lowercase-slug>` (for example
`8-contrib-pr-workflow`). One branch per issue.

## Step 3 — Commit locally

- Conventional Commits, one logical change per commit.
- Keep the diff inside the issue scope.
- Run the checks for the touched area (build, `ctest`, `clang-format --dry-run
  --Werror`, `clang-tidy -p <build-dir>`).

## Step 4 — Open the PR

```bash
git push -u origin "N-<slug>"      # or the fork remote
gh pr create --repo PennNonchalant/RayLiDAR --base main --head "N-<slug>" \
  --title "<type>: <summary> (#N)" \
  --body-file <body.md> \
  --assignee @me
```

The body must fill `.github/pull_request_template.md` (Description, New library
used, Type of change, Additional notes) and contain `Closes #N`.

Then stop and wait. Ask for a reviewer if one is not assigned automatically:

```bash
gh pr edit <pr> --add-reviewer <login>
```

## Step 5 — Review and merge (reviewer only)

- Check CI is green and the diff matches the issue.
- Approve, then squash merge and delete the branch.
- The author rebases/updates on request; the author does not self-merge.

## Guardrails

- Refuse to run `git push origin main` or `git push -f`.
- If the current branch is `main`, stop and create a branch.
- If no issue number can be determined, ask the user before creating a branch.
- Never commit `.planning/`, `docs-local/`, `.git/`, or `AGENTS.md` (ignored local files).
- Do not mix unrelated changes in the PR; open a second issue instead.
