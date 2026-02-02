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
  export GIT_TERMINAL_PROMPT=0
  git -c credential.helper= -c credential.helper="!gh auth git-credential" push
else
  export GIT_TERMINAL_PROMPT=0
  git -c credential.helper= -c credential.helper="!gh auth git-credential" push -u origin "$branch"
fi

base_branch="${BASE_BRANCH:-main}"
pr_title="${PR_TITLE:-$branch}"
pr_body_file="${PR_BODY_FILE:-}"
pr_labels="${PR_LABELS:-}"

pr_exists=false
if gh pr view "$branch" >/dev/null 2>&1; then
  pr_exists=true
fi

if [ "$pr_exists" = false ]; then
  if [ -n "$pr_body_file" ] && [ -f "$pr_body_file" ]; then
    gh pr create --base "$base_branch" --head "$branch" --title "$pr_title" --body-file "$pr_body_file"
  else
    gh pr create --base "$base_branch" --head "$branch" --title "$pr_title" --body "Automated publisher PR for $branch"
  fi
fi

if [ -n "$pr_labels" ]; then
  IFS=',' read -r -a labels <<< "$pr_labels"
  for label in "${labels[@]}"; do
    trimmed=$(echo "$label" | xargs)
    if [ -n "$trimmed" ]; then
      gh pr edit "$branch" --add-label "$trimmed" >/dev/null 2>&1 || true
    fi
  done
fi

if [ "$pr_exists" = true ]; then
  echo "PR already exists for $branch."
fi
