#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
for item in "control:$CONTROL" "prism:$PRISM"; do
  label=${item%%:*}; run=${item#*:}; pid=$(cat "$run/runtime/wrapper.pid")
  kill -0 "$pid" 2>/dev/null && alive=yes || alive=no
  progress=$(tr '\r' '\n' < "$run/runtime/driver.log" | grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint' | tail -n1 || true)
  fatal=$(grep -Eic 'Traceback|out of memory|CUDA error|RayActorError|worker died|ErrorInitializationFailed|NCCL.*error' "$run/runtime/driver.log" || true)
  printf '%s alive=%s fatal=%s progress=%s\n' "$label" "$alive" "$fatal" "$progress"
done
echo 'prism_local_shards:'
find "$PRISM" -path '*/global_step_*/actor/local_shard_checkpoint/checkpoint_rank_*.pt' -printf '%P %s bytes\n' | sort -V
