#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
V1=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1
V2=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
HEAD=554c6dc8d586162d9444c01fa88308ed4f5203d0
BRANCH=codex/sz-7d07a421-grpo-pi0-robotwin

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
id
printf 'head=%s\nbranch=%s\nstatus_begin\n' \
  "$(git -C "$WT" rev-parse HEAD)" "$(git -C "$WT" branch --show-current)"
git -C "$WT" status --short
printf '%s\n' status_end
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$WT" status --porcelain)"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$HEAD"
test -s "$V1/resolved.yaml"
test ! -e "$V2"
test ! -e "$PACKET"

printf '%s\n' 'ray_processes_begin'
pgrep -a -u "$(id -u)" -f '(^|/)(raylet|gcs_server)( |$)' || true
printf '%s\n' 'ray_processes_end'
if pgrep -u "$(id -u)" -x raylet >/dev/null || pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  printf '%s\n' 'unexpected live Ray cluster for chenyiteng' >&2
  exit 1
fi

printf '%s\n' 'old_grpo_processes_begin'
pgrep -a -u "$(id -u)" -f '[t]rain_embodied_agent.py.*robotwin_adjust_bottle_grpo_openpi' || true
printf '%s\n' 'old_grpo_processes_end'
if pgrep -u "$(id -u)" -f '[t]rain_embodied_agent.py.*robotwin_adjust_bottle_grpo_openpi' >/dev/null; then
  printf '%s\n' 'unexpected live GRPO driver' >&2
  exit 1
fi

printf '%s\n' 'gpu_summary_begin'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' 'gpu_summary_end'
printf '%s\n' 'gpu_4_7_compute_begin'
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
printf '%s\n' 'gpu_4_7_compute_end'
if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-7 are not idle' >&2
  exit 1
fi

printf '%s\n' 'fastwam_processes_begin'
pgrep -a -u "$(id -u)" -f 'FastWAM|fastwam|eval_policy.*move_stapler_pad|eval_policy.*turn_switch|eval_policy.*pick_diverse_bottles' || true
printf '%s\n' 'fastwam_processes_end'

awk '/^MemTotal:|^MemAvailable:|^SwapTotal:|^SwapFree:/ {print}' /proc/meminfo
df -h / /home /data
df -Pk /data
printf '%s\n' 'SZ_GRPO_RETRY_V2_LIVE_PREFLIGHT_OK'
