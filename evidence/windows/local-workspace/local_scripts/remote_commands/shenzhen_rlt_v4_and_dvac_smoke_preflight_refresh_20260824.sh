#!/usr/bin/env bash
set -euo pipefail

RLT=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
RUNTIME="$RLT/runtime"
LOG="$RUNTIME/driver.log"
CKPT="$RLT/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3/checkpoints"
DVAC=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current

printf 'MARKER=SZ_RLT_V4_DVAC_SMOKE_PREFLIGHT_REFRESH_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds

printf '%s\n' '=== RLT LIVE ==='
pid=$(cat "$RUNTIME/wrapper.pid" 2>/dev/null || true)
printf 'wrapper_pid=%s\n' "$pid"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  printf 'wrapper_alive=yes\n'
else
  printf 'wrapper_alive=no\n'
fi
if [[ -e "$RUNTIME/exit_code.txt" ]]; then
  printf 'exit_code='; cat "$RUNTIME/exit_code.txt"
else
  printf 'exit_code=pending\n'
fi
grep -aE 'Global Step:|update_step|eval|success|Traceback|OutOfMemory|WorkerCrashed|ERROR' "$LOG" 2>/dev/null | tail -n 45 || true
find "$CKPT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -Vu | tail -n 12

printf '%s\n' '=== DVAC SOURCE ==='
git -C "$DVAC" status --short --branch
printf 'head=%s\n' "$(git -C "$DVAC" rev-parse HEAD)"
printf 'remote=%s\n' "$(git -C "$DVAC" rev-parse personal/codex/sz-current-pi0-dvac-grpo)"

printf '%s\n' '=== LIVE RESOURCE ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'gpu0_3_compute_pids_begin\n'
nvidia-smi -i 0,1,2,3 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits || true
printf 'gpu0_3_compute_pids_end\n'
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
df -h / /home /data
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray status | sed -n '1,30p'
printf 'MARKER=SZ_RLT_V4_DVAC_SMOKE_PREFLIGHT_REFRESH_OK\n'
