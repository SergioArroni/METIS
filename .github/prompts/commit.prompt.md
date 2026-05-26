---
description: "Automate git add, conventional commit, pre-commit hooks, and push. Use when you want to stage, commit, and push changes following Conventional Commits standard."
agent: "agent"
argument-hint: "Describe the changes or specify a target branch"
---

# Git Commit & Push

Perform the full git commit and push workflow for the current workspace.

## Steps

1. **Stage changes**: Run `git add -A` to stage all changes. If the user specifies particular files, stage only those.

2. **Review staged changes**: Run `git diff --cached --stat` and `git diff --cached` to understand what's being committed. If there are no staged changes, inform the user and stop.

3. **Validate before commit**:
   - If `.pre-commit-config.yaml` exists, run `python -m pre_commit run --all-files`.
   - If there are obvious validation commands relevant to the changed files, run them too.
   - If validation fails, stop before commit, show the errors, and fix them if the failure is straightforward and directly related to the staged changes.
   - Do not commit while validation is red.

4. **Generate a conventional commit message** based on the diff:
   - Format: `<type>(<scope>): <descripción>`
   - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
   - Scope: infer from the changed files (optional but preferred)
   - Description: concise, imperative mood, lowercase, no trailing period
   - Body: if changes are complex, add a body after a blank line explaining the "why"
   - Breaking changes: add `BREAKING CHANGE:` footer if applicable
   - First line must be under 72 characters
   - **IMPORTANT**: Always write the description and body in Spanish

5. **Show the commit message** to the user and proceed.

6. **Commit**: Run `git commit` with the generated message. Pre-commit hooks will execute automatically if configured in the repo.

7. **Handle pre-commit failures**: If hooks fail, show the errors and stop. Do not bypass hooks unless the user explicitly asks.

8. **Push**:
   - If the user specifies a target branch → `git push origin <branch>`
   - Otherwise → `git push origin HEAD` (current branch)
   - If the upstream is not set → `git push --set-upstream origin <branch>`
   - Never force push unless explicitly requested.

## User Input

{{input}}
