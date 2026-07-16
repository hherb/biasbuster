---
name: fixall
description: Use when a code review has produced issues that need to be addressed and the pull request finalized on the biasbuster project.
disable-model-invocation: true
allowed-tools: Read, Edit, Bash(git add *), Bash(git commit *), Bash(git push *), Bash(git status *), Bash(git diff *), Bash(gh issue *), Bash(gh pr *), Bash(uv run pytest *)
---

Address all issues identified in the code review one by one. If fixing them appears manageable within this session, fix them now. If not, lodge the issue on GitHub. Once all issues have been addressed, run the test suite with `uv run pytest` and review the code changes thoroughly against CLAUDE.md's coding standards ("STOP AND CHECK" and "Python Coding Standards"). If satisfied no issues are left open, update HANDOVER.md and ROADMAP.md ONLY if necessary to reflect these changes. Then stage the files you changed explicitly (never `git add -A` / `git add .`), commit, and push the changes into the PR.
