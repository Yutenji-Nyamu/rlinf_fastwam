#!/usr/bin/env bash
set -euo pipefail

staging=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/staging
if [[ -e "$staging" ]]; then
  echo "staging already exists: $staging" >&2
  exit 1
fi
mkdir -p "$staging/docs/rlinf-robotwin-pi0-rltoken/evidence"
printf 'staging=%s\n' "$staging"
