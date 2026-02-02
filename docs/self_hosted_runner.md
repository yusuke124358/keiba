# Self-Hosted Runner Setup (Codex CLI)

This repo assumes Codex CLI runs on a self-hosted runner machine with ChatGPT Pro login
(subscription-based) and **no OPENAI_API_KEY in GitHub Actions**.

## Install Codex CLI
Install Node/npm and Codex CLI on the runner:

```bash
npm i -g @openai/codex
```

## Login (one-time, headless)
On the runner:

```bash
codex login --device-auth
```

Follow the device code flow in a browser to complete.

**Important:** The authentication cache (example: `~/.codex/auth.json`) is equivalent to a password.
Do not commit it or copy it into the repo. Protect it on the runner.

## Verify
```bash
codex login status
codex --version
```

Exit code 0 for `codex login status` means the runner is ready.

## Workflow Config
- Workflows copy `.github/runner/codex.config.toml` to `~/.codex/config.toml` on each run.
- Keep that config free of secrets; it only enforces login method and defaults.

## Safety
- Only run this automation on **trusted PRs** (no forks).
- This repo enforces fork PR exclusion in `scripts/agent/review_loop.py`.
