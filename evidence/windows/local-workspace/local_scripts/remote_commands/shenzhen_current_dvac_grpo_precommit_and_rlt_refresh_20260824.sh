#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RLT=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix

printf 'MARKER=SZ_CURRENT_DVAC_GRPO_PRECOMMIT_RLT_REFRESH_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
printf '%s\n' '=== DVAC WORKTREE ==='
git -C "$WT" status --short --branch
git -C "$WT" diff --cached --check
git -C "$WT" diff --cached --stat
source "$VENV/bin/activate"
if command -v ruff >/dev/null 2>&1; then
  ruff check \
    "$WT/rlinf/algorithms/dvac_train_weighting.py" \
    "$WT/rlinf/workers/actor/embodied_fsdp_actor_worker.py" \
    "$WT/rlinf/workers/rollout/hf/huggingface_worker.py" \
    "$WT/tests/unit_tests/test_dvac_train_weighting.py"
else
  printf 'ruff=not-installed\n'
fi

printf '%s\n' '=== RLT LIVE ==='
pid=$(cat "$RLT/driver.pid" 2>/dev/null || true)
printf 'driver_pid=%s\n' "$pid"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then printf 'driver_alive=yes\n'; else printf 'driver_alive=no\n'; fi
grep -aE 'Global Step:|update_step|fixed|success|Traceback|OutOfMemory|WorkerCrashed' "$RLT/driver.log" 2>/dev/null | tail -n 35 || true
find "$RLT" -path '*/checkpoints/global_step_*' -maxdepth 6 -type d -printf '%f\n' 2>/dev/null | sort -Vu | tail -n 15

printf '%s\n' '=== RESOURCE ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
df -h / /home /data
printf 'MARKER=SZ_CURRENT_DVAC_GRPO_PRECOMMIT_RLT_REFRESH_OK\n'
