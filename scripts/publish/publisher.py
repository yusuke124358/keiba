#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return Path(out)


def run(cmd, cwd=None, env=None) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="main")
    p.add_argument("--labels", default="autogen,auto-fix")
    p.add_argument("--title", required=True)
    p.add_argument("--body-file", required=True)
    args = p.parse_args()

    root = repo_root()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not branch:
        raise RuntimeError("Unable to determine current branch.")
    if branch == args.base:
        raise RuntimeError("Refusing to publish from base branch.")

    env = os.environ.copy()
    token = env.get("AUTO_FIX_PUSH_TOKEN") or env.get("GH_TOKEN")
    if token:
        env["GH_TOKEN"] = token
    run(["git", "push", "origin", f"HEAD:{branch}"], cwd=root, env=env)
    run(
        [
            "gh",
            "pr",
            "create",
            "--base",
            args.base,
            "--head",
            branch,
            "--title",
            args.title,
            "--body-file",
            args.body_file,
        ],
        cwd=root,
        env=env,
    )
    if args.labels:
        run(
            ["gh", "pr", "edit", branch, "--add-label", args.labels],
            cwd=root,
            env=env,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
