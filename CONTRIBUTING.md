# Contributing to RayLiDAR

`main` is protected. Every change reaches it through an issue, a branch, and a
reviewed pull request — in that order. Do not commit or push directly to `main`.

The short version: **announce the change in an issue, create a branch for it,
open a PR when it is ready, and wait for someone else to review and merge.**

## 1. Declare the change with an issue

Create one issue per change before writing code.

- Title: `<area>: <short description>` (for example `docs: document the PR workflow`).
- Body: background, scope, and acceptance criteria. Say what will change and how
  the change will be validated.
- Assign the issue to yourself.
- Link related issues or PRs.

```bash
gh issue create --assignee @me \
  --title "docs: document the PR workflow" \
  --body "## Background
...
## Scope
...
## Acceptance criteria
- [ ] ..."
```

## 2. Create the branch

Branch from the current `main`; never branch from an outdated local copy.

- Name: `<issue-number>-<slug>`, for example `8-contrib-pr-workflow`.
- Keep one branch per issue.

```bash
git fetch origin
git switch -c 8-contrib-pr-workflow origin/main
```

If your account has no write access to the repository, push the branch to your
fork and open the PR from there (`<you>:<branch>` → `PennNonchalant:main`).

## 3. Make the change locally

- Keep commits small and scoped; the repository uses Conventional Commits
  (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`).
- Do not mix unrelated edits in one branch.
- Run the checks the change touches before opening the PR:

```bash
cmake --preset linux-debug -B build/linux-debug
cmake --build build/linux-debug -j
ctest --test-dir build/linux-debug --output-on-failure
clang-format --dry-run --Werror --style=file $(git ls-files "*.cpp" "*.h")
clang-tidy -p build/linux-debug $(git ls-files "*.cpp")
```

## 4. Open the pull request

Push the branch and open a PR against `main`.

- Fill in every section of the PR template (`.github/pull_request_template.md`),
  including the "New library used" section.
- Reference the issue in the body with `Closes #<issue-number>` so it closes on
  merge.
- Keep the PR focused on the issue scope; note follow-up work separately.
- Wait for CI and for review.

```bash
git push -u origin 8-contrib-pr-workflow
gh pr create --base main --head 8-contrib-pr-workflow \
  --title "docs: document the PR workflow (#8)" \
  --body-file pr-body.md
```

## 5. Review and merge

- The author does not merge their own PR without an approving review.
- Address review comments with new commits on the same branch (do not force-push
  over someone else's review unless asked).
- Merge only when CI is green and the review is approved; prefer squash merge.
- Delete the branch after merge, then update your local copy:

```bash
git switch main
git pull --ff-only
git branch -d 8-contrib-pr-workflow
```

## Cross-platform notes

- **Line endings:** the repository is stored as LF and `.gitattributes` enforces
  it. Do not set `core.autocrlf=true` for this repository; if your working copy
  already has CRLF, renormalize it with `git add --renormalize .`. This keeps
  `clang-format` (`LineEnding: LF`) identical on Windows and Linux.
- **Formatting/lint:** results depend on tool versions. `clang-format` and
  `clang-tidy` 18+ are expected; run them locally on Linux, because CI only
  builds on Windows.
- **Vulkan:** the smoke test requires a Vulkan 1.4 loader. Ubuntu 25.04+ ships
  one (1.4.304); Ubuntu 24.04 ships 1.3.275 and will fail even with a 1.4 driver.
- **Build requirements:** CMake 3.29+, Ninja, and vcpkg with the baseline pinned
  in `vcpkg.json`.

## Hard rules

- No direct pushes to `main`.
- No PR without an issue.
- No merge without review, even for maintainers.
- No unrelated changes in the same PR.
