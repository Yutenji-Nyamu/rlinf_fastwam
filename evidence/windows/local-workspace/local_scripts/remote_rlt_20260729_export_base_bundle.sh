#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
out=/root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle

if test -e "$out"; then
  echo "refusing: bundle already exists: $out" >&2
  exit 10
fi

git -C "$repo" bundle create "$out" codex/rlt-pi0-robotwin
git -C "$repo" bundle verify "$out"
stat -c '%s %n' "$out"
sha256sum "$out"
