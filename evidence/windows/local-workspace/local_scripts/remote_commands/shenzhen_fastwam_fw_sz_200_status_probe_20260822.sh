#!/usr/bin/env bash
set -euo pipefail
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
CUROBO="$FW/third_party/RoboTwin/envs/curobo"
printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
git -C "$FW" status --porcelain --untracked-files=normal
printf 'curobo_head=%s\n' "$(git -C "$CUROBO" rev-parse HEAD)"
git -C "$CUROBO" status --short --branch
for p in "$FW/third_party/RoboTwin/assets" "$FW/third_party/RoboTwin/task_config" "$FW/third_party/RoboTwin/policy/fastwam_policy"; do
  if test -e "$p" || test -L "$p"; then printf 'present=%s\n' "$p"; else printf 'absent=%s\n' "$p"; fi
done
