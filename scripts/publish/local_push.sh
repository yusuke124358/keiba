#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "main" ]; then
  echo "Refusing to publish from main. Checkout a feature branch."
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install GitHub CLI and authenticate before publishing."
  exit 2
fi

gh auth status >/dev/null 2>&1 || {
  echo "gh auth status failed. Run: gh auth login";
  exit 2;
}

if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  git -c credential.helper= -c credential.helper="!gh auth git-credential" push
else
  git -c credential.helper= -c credential.helper="!gh auth git-credential" push -u origin "$branch"
fi

base_branch="${BASE_BRANCH:-main}"
pr_title="${PR_TITLE:-$branch}"
pr_body_file="${PR_BODY_FILE:-}"

if gh pr view "$branch" >/dev/null 2>&1; then
  echo "PR already exists for $branch."
  exit 0
fi

if [ -n "$pr_body_file" ] && [ -f "$pr_body_file" ]; then
  gh pr create --base "$base_branch" --head "$branch" --title "$pr_title" --body-file "$pr_body_file"
else
  gh pr create --base "$base_branch" --head "$branch" --title "$pr_title" --body "Automated publisher PR for $branch"
fi
