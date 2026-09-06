#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
CONTROL=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2

echo '=== wrappers ==='
for item in prism:$RUN control:$CONTROL; do
  name=${item%%:*}; path=${item#*:}; pid=$(cat "$path/runtime/wrapper.pid")
  printf '%s pid=%s alive=' "$name" "$pid"; kill -0 "$pid" 2>/dev/null && echo yes || echo no
done
echo '=== resolved essentials ==='
grep -nE 'component_placement:|max_steps: 100$|val_check_interval:|save_interval:|adv_type:|filter_rewards:|total_num_envs:|rollout_epoch:|global_batch_size:|micro_batch_size:|checkpoint_format:|selected_l:|quality_lambda:' "$RUN/runtime/resolved.yaml" | head -n 80
echo '=== log tail ==='
tail -n 80 "$RUN/runtime/driver.log" 2>/dev/null || true
echo '=== fatal ==='
grep -Ei 'traceback|out of memory|CUDA error|worker died|RayActorError|ErrorInitializationFailed|NCCL.*error' "$RUN/runtime/driver.log" 2>/dev/null | tail -n 20 || true
echo '=== GPUs/RAM ==='
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/^MemAvailable:/ {print}' /proc/meminfo
echo '=== Ray jobs/actors ==='
ps -eo pid,user,args | grep -E 'ray::(Actor|EnvWorker|RolloutWorker)|train_embodied_agent.py' | grep -v grep | tail -n 40 || true
