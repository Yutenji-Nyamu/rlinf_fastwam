#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo/runs/ppo-dvac-action-adv-fix-w0to2-smoke2-2gpu64x4-b1024-noeval-localshard-phys23-v1
RUNTIME=$ROOT/runtime
EXP=$ROOT/robotwin_ppo_dvac_action_adv_fix_w0to2_smoke2_2gpu64x4_b1024_noeval_localshard_phys23_v1
CKPT=$EXP/checkpoints/global_step_2

TZ=Asia/Shanghai date --iso-8601=seconds
printf 'EXIT='; cat "$RUNTIME/exit_code.txt"
printf 'WRAPPER='; cat "$RUNTIME/wrapper.pid"
if kill -0 "$(cat "$RUNTIME/wrapper.pid")" 2>/dev/null; then echo ALIVE; else echo DEAD; fi

echo FATAL_SCAN
grep -aEn 'Traceback|Error executing job|CUDA out of memory|OutOfMemory|RayActorError|WorkerCrashedError|nan|inf' "$RUNTIME/driver.log" | tail -n 40 || true

echo METRIC_LINES
grep -aE 'Global Step:|actor/dvac_weight|actor/grad_norm|actor/policy_loss|critic/|value_loss|explained_variance|Saving checkpoint|Saved checkpoint' "$RUNTIME/driver.log" | tail -n 120 || true

echo CHECKPOINT_FILES
find "$CKPT" -maxdepth 5 \( -type f -o -type l \) -printf '%y %s %P\n' | sort
echo CHECKPOINT_TOTAL
du -sh "$CKPT"

echo GPU
nvidia-smi -i 2,3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo OWNED_PROCESSES
pgrep -af 'ppo-dvac-action-adv-fix-w0to2-smoke2-2gpu64x4-b1024-noeval-localshard-phys23-v1|robotwin_ppo_dvac_action_adv_fix_w0to2_smoke2' || true

echo DONE
