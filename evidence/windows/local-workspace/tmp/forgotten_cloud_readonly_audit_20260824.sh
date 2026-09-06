#!/usr/bin/env bash
set -euo pipefail

audit_repo() {
  label=$1
  repo=$2
  printf '\n=== %s ===\n' "$label"
  if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'NOT_GIT %s\n' "$repo"
    return
  fi
  printf 'path=%s\nbranch=%s\nhead=%s\n' \
    "$repo" \
    "$(git -C "$repo" branch --show-current)" \
    "$(git -C "$repo" rev-parse HEAD)"
  printf 'status\n'
  git -C "$repo" status --short --branch
  upstream=$(git -C "$repo" rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || true)
  printf 'upstream=%s\n' "${upstream:-NONE}"
  if [ -n "$upstream" ]; then
    printf 'ahead_behind='; git -C "$repo" rev-list --left-right --count "$upstream...HEAD"
  fi
  printf 'recent\n'
  git -C "$repo" log -3 --oneline --decorate
  printf 'remote_tracking_same_name\n'
  branch=$(git -C "$repo" branch --show-current)
  git -C "$repo" for-each-ref \
    --format='%(refname:short)|%(objectname)' \
    "refs/remotes/*/$branch"
}

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
audit_repo dvac-inference /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
audit_repo robotwin-control-trace /root/autodl-tmp/idea2_dvac_train_wamppo
audit_repo old-main /root/autodl-tmp/RLinf

printf '\n=== OLD_MAIN_UNTRACKED_FILES ===\n'
git -C /root/autodl-tmp/RLinf ls-files --others --exclude-standard -z \
  | while IFS= read -r -d '' relative; do
      absolute=/root/autodl-tmp/RLinf/$relative
      if [ -f "$absolute" ]; then
        stat -c '%s bytes | %n' "$absolute"
      elif [ -L "$absolute" ]; then
        printf 'symlink | %s -> %s\n' "$absolute" "$(readlink "$absolute")"
      else
        printf 'other | %s\n' "$absolute"
      fi
    done

printf '\n=== ACTIVE_GLOBAL_Z ===\n'
run=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
for name in wrapper driver observer; do
  pid=$(cat "$run/$name.pid" 2>/dev/null || true)
  printf '%s=%s alive=%s\n' "$name" "$pid" "$([ -n "$pid" ] && [ -d "/proc/$pid" ] && echo 1 || echo 0)"
done
grep -a 'Global Step' "$run/driver.log" 2>/dev/null | tail -n 3 || true
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
grep -E '^oom |^oom_kill ' /sys/fs/cgroup/memory.events
