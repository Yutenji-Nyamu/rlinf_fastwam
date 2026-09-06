#!/usr/bin/env bash
set -u

raylogs=/data/chenyiteng/ray/rlt-dsrl-v3/session_latest/logs
worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421

for pid in 391536 391538; do
  for file in "$raylogs"/*-"$pid".err "$raylogs"/*-"$pid".out; do
    [ -f "$file" ] || continue
    printf '### %s\n' "$file"
    nl -ba "$file"
  done
done

printf '%s\n' '--- checkpoint symbols ---'
grep -R -n --include='*.py' -E 'def save_checkpoint|async def save_checkpoint|dcp_checkpoint|save_full_model_weights' \
  "$worktree/rlinf" | head -n 240 || true

printf '%s\n' '--- RLT worker method context ---'
file=$(grep -R -l --include='*.py' 'class RLTACFSDPPolicy' "$worktree/rlinf" | head -n 1)
printf 'file=%s\n' "$file"
if [ -n "$file" ]; then
  grep -n -A180 -B40 'def save_checkpoint' "$file" | head -n 260 || true
fi

