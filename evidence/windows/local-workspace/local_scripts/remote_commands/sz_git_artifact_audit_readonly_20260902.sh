#!/usr/bin/env bash
set -u

printf 'AUDIT_TIME='; date '+%F %T %Z'
root=/data/chenyiteng/projects/rlinf-shenzhen

printf '\n=== RLINF_WORKTREES ===\n'
for repo in "$root"/RLinf "$root"/worktrees/*; do
  test -d "$repo" || continue
  git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1 || continue
  printf '\nREPO=%s\n' "$repo"
  printf 'HEAD='; git -C "$repo" rev-parse HEAD 2>/dev/null || true
  printf 'BRANCH='; git -C "$repo" symbolic-ref --quiet --short HEAD 2>/dev/null || printf 'DETACHED\n'
  printf 'UPSTREAM='; git -C "$repo" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || printf 'NONE\n'
  printf 'UPSTREAM_HEAD='; git -C "$repo" rev-parse '@{u}' 2>/dev/null || printf 'NONE\n'
  printf 'AHEAD_BEHIND='; git -C "$repo" rev-list --left-right --count '@{u}...HEAD' 2>/dev/null || printf 'NA\n'
  status="$(git -C "$repo" status --porcelain=v1 --untracked-files=all 2>/dev/null || true)"
  if test -z "$status"; then
    printf 'DIRTY_COUNT=0\n'
  else
    printf 'DIRTY_COUNT=%s\n' "$(printf '%s\n' "$status" | wc -l)"
    printf '%s\n' "$status" | sed -n '1,12p' | sed 's/^/STATUS=/'
  fi
  git -C "$repo" remote -v 2>/dev/null | sed -n '1,6p' | sed 's/^/REMOTE=/'
done

printf '\n=== ACTUAL_PERSONAL_REMOTE_HEADS ===\n'
probe_repo="$root/worktrees/pi05-robotwin-rl"
if test -d "$probe_repo"; then
  timeout 25s git -C "$probe_repo" ls-remote --heads personal 'refs/heads/codex/*' 2>&1 || true
fi

printf '\n=== OFFICIAL_FASTWAM_SOURCE ===\n'
for repo in /data/chenyiteng/projects/fastwam-standalone/FastWAM-* /data/chenyiteng/projects/fastwam-standalone/FastWAM-*/*; do
  test -d "$repo/.git" || continue
  printf '\nREPO=%s\n' "$repo"
  printf 'HEAD='; git -C "$repo" rev-parse HEAD 2>/dev/null || true
  printf 'BRANCH='; git -C "$repo" symbolic-ref --quiet --short HEAD 2>/dev/null || printf 'DETACHED\n'
  printf 'DIRTY_COUNT='; git -C "$repo" status --porcelain=v1 --untracked-files=all 2>/dev/null | wc -l
  git -C "$repo" status --porcelain=v1 --untracked-files=all 2>/dev/null | sed -n '1,12p' | sed 's/^/STATUS=/'
  git -C "$repo" remote -v 2>/dev/null | sed -n '1,6p' | sed 's/^/REMOTE=/'
done

printf '\n=== CURRENT_RUN_CANDIDATES ===\n'
find /data/chenyiteng/results/rlinf-shenzhen -mindepth 3 -maxdepth 5 -type d \
  \( -name '*fastwam-grpo-control-formal100-2gpu32x8*' -o -name '*pi05-grpo-control-formal100-2gpu64x4*g8*b1024*u2*' \) \
  -print 2>/dev/null | sort -u

printf '\n=== CURRENT_RUN_SMALL_ARTIFACTS ===\n'
for run in \
  /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2 \
  /data/chenyiteng/results/rlinf-shenzhen/pi05/runs/*control*formal100*2gpu64x4*g8*b1024*u2*; do
  test -d "$run" || continue
  printf '\nRUN=%s\n' "$run"
  du -sh "$run" 2>/dev/null | sed 's/^/TOTAL=/'
  for sub in runtime tensorboard checkpoints video robotwin_data monitor telemetry figures; do
    test -e "$run/$sub" || continue
    du -sh "$run/$sub" 2>/dev/null | sed "s#^#SUB_${sub}=#"
  done
  find "$run" -type f \
    \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.txt' -o -name '*.jsonl' -o -name 'events.out.tfevents*' -o -name '*.png' \) \
    -printf '%s\t%p\n' 2>/dev/null | sort -nr | sed -n '1,35p' | sed 's/^/SMALL_CANDIDATE=/'
done
