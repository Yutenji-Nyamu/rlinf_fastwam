#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
RLINF=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
FASTWAM=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
KEY=/home/chenyiteng/.ssh/github_rlinf_fastwam_deploy_ed25519
KNOWN=/home/chenyiteng/.ssh/github_rlinf_fastwam_known_hosts

printf 'MARKER=SZ_IMPLEMENTATION_PREFLIGHT_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

printf '%s\n' '=== PPO ==='
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)
printf 'driver_pid=%s\n' "$DRIVER"
if [[ -n "$DRIVER" ]] && kill -0 "$DRIVER" 2>/dev/null; then
  printf 'driver_alive=yes\n'
else
  printf 'driver_alive=no\n'
fi
grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$RUN/driver.log" | tail -n 28 || true
printf 'fatal_count='
grep -aiEc 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left|NCCL.*(error|failed)' "$RUN/driver.log" || true
printf 'checkpoints='
find "$RUN/robotwin_ppo_openpi/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f ' 2>/dev/null | sort -V
printf '\n'

printf '%s\n' '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true

printf '%s\n' '=== MEMORY ==='
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
if [[ -n "$DRIVER" ]] && [[ -r "/proc/$DRIVER/cgroup" ]]; then
  CGREL=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  CG=/sys/fs/cgroup${CGREL}
  printf 'driver_cgroup=%s\n' "$CG"
  for f in memory.current memory.peak memory.swap.current memory.events; do
    if [[ -r "$CG/$f" ]]; then
      printf -- '-- %s --\n' "$f"
      cat "$CG/$f"
    fi
  done
fi

printf '%s\n' '=== RLINF GIT ==='
git -C "$RLINF" rev-parse HEAD
git -C "$RLINF" status --short
git -C "$RLINF" remote -v
git -C "$RLINF" worktree list --porcelain
for branch in codex/sz-7d07a421-grpo-pi0-robotwin codex/sz-current-pi0-dvac-observe; do
  if git -C "$RLINF" show-ref --verify --quiet "refs/heads/$branch"; then
    printf 'local_branch_exists=%s\n' "$branch"
  else
    printf 'local_branch_absent=%s\n' "$branch"
  fi
done
for path in \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421 \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421; do
  if [[ -e "$path" ]]; then
    printf 'planned_path_exists=%s\n' "$path"
  else
    printf 'planned_path_absent=%s\n' "$path"
  fi
done

printf '%s\n' '=== FASTWAM GIT ==='
git -C "$FASTWAM" rev-parse HEAD
git -C "$FASTWAM" status --short
git -C "$FASTWAM" remote -v
git -C "$FASTWAM" worktree list --porcelain
if git -C "$FASTWAM" show-ref --verify --quiet refs/heads/codex/sz-fastwam-dvac-observe; then
  printf 'local_branch_exists=codex/sz-fastwam-dvac-observe\n'
else
  printf 'local_branch_absent=codex/sz-fastwam-dvac-observe\n'
fi
FW_WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
if [[ -e "$FW_WT" ]]; then
  printf 'planned_path_exists=%s\n' "$FW_WT"
else
  printf 'planned_path_absent=%s\n' "$FW_WT"
fi

printf '%s\n' '=== SSH MATERIAL ==='
stat -c '%n mode=%a bytes=%s' "$KEY" "$KEY.pub"
ssh-keygen -lf "$KEY.pub" -E sha256
if [[ -e "$KNOWN" ]]; then
  stat -c '%n mode=%a bytes=%s' "$KNOWN"
else
  printf 'known_hosts_absent=%s\n' "$KNOWN"
fi

printf '%s\n' '=== DISK ==='
df -h / /home /data
printf 'MARKER=SZ_IMPLEMENTATION_PREFLIGHT_OK\n'

