#!/usr/bin/env bash
set -u

printf '%s\n' '--- DSRL worker cwd ---'
for pid in $(pgrep -u "$(id -u)" -f 'ray::(EnvWorker|MultiStepRolloutWorker|EmbodiedSACFSDPPolicy)' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  printf 'pid=%s cwd=%s cmd=%s\n' "$pid" "$cwd" "$cmd"
done
printf '%s\n' '--- likely relative data roots ---'
for path in \
  /home/chenyiteng/data \
  /data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin/data \
  /data/chenyiteng/ray/rlt-dsrl-v3/session_latest/data; do
  if [ -e "$path" ]; then
    printf 'exists %s\n' "$path"
    find "$path" -maxdepth 2 -type f -mmin -20 -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | head -n 30
  else
    printf 'absent %s\n' "$path"
  fi
done
