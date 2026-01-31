# Codex team config (profiles, rules, memory)

## Trusted project
- `.codex/config.toml` is read only when the project is trusted.
- If settings do not apply, verify the project is trusted in Codex CLI.

## Profiles (summary)
Default (no profile):
- `approval_policy = on-request`
- `sandbox_mode = read-only`
- `web_search = disabled`
- `project_doc_fallback_filenames = ["memory.md"]`

Profiles:
- `orchestrator`: workspace-write, web_search disabled
- `experiment`: workspace-write, web_search disabled
- `reviewer`: read-only, web_search disabled
- `research`: read-only, web_search live
- `publisher`: workspace-write, web_search disabled

## Network access (workspace-write)
- `sandbox_workspace_write.network_access = false` by default.
- This blocks outbound network in workspace-write sandboxes.
- If you need to push or create PRs from a sandboxed profile, either:
  - run those commands outside Codex, or
  - temporarily override network access for the run.

## Example usage
```bash
codex exec --profile research --search "paper title or query"
codex exec --profile experiment --full-auto "run backtest and summarize"
codex exec --profile reviewer "/review"
```

Note:
- `--full-auto` is treated as a shortcut for on-request approval with
  workspace-write in this repo's policy.

## Rules (exec policy)
- Rules are stored in `.codex/rules/keiba.rules`.
- Destructive commands are forbidden; external-impact commands are prompted.

Validate rules:
```bash
codex execpolicy check --pretty --rules .codex/rules/keiba.rules -- git status
codex execpolicy check --pretty --rules .codex/rules/keiba.rules -- git push
```

On Windows, if `codex` is blocked by execution policy, use:
```bash
codex.cmd execpolicy check --pretty --rules .codex/rules/keiba.rules -- git status
```
