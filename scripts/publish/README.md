# Publisher Modes

## Local publisher (human-authenticated)
Use when a developer has already authenticated GitHub CLI locally.

```bash
bash scripts/publish/local_push.sh
```

Optional env vars:
- `BASE_BRANCH` (default: `main`)
- `PR_TITLE` (default: branch name)
- `PR_BODY_FILE` (path to PR body)
- `PR_LABELS` (comma-separated labels)

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish\local_push.ps1 -Labels "autogen,auto-fix"
```

## GitHub Actions publisher
The workflow applies patches from `artifacts/patches/*.diff`, commits, and opens a PR.
Trigger with `workflow_dispatch` or schedule.

Patch format: unified diff created by `git diff` (no commit metadata required).
Ensure patch files are committed to the repo before triggering the workflow.
