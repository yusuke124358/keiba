# Windows / WSL Notes

## Recommended setup
- Prefer WSL2 for development.
- Keep the repo under `/home/<user>/...` rather than `/mnt/c/...` for stability.

## Sandbox networking
- Network access may be disabled in sandboxed environments by default.
- Use cached web search first; request live access only when required.

## Git push reliability
- Git push may hang or fail when interactive auth prompts appear.
- Use PAT or SSH keys configured for non-interactive auth.
- Publisher scripts are the default path; Codex should not push directly.

## /apps integration
- `/apps` helpers can supply GitHub context, but are not a full push replacement.
- Use them to assist publisher setup, not as the sole release path.
