#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo/runs
CONTROL=$ROOT/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1
DVAC=$ROOT/ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
CODE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix

date --iso-8601=seconds
echo '=== immutable source identity ==='
for item in "control:$CONTROL" "dvac:$DVAC"; do
  label=${item%%:*}; run=${item#*:}
  echo "--- $label ---"
  printf 'source_head='; cat "$run/runtime/source_head.txt"
  printf 'exit_code='; cat "$run/runtime/exit_code.txt" 2>/dev/null || echo pending
  printf 'finished_at='; cat "$run/runtime/finished_at.txt" 2>/dev/null || echo pending
  printf 'contract='; cat "$run/runtime/contract.json"; echo
done
echo "code_root=$CODE"
echo "head=$(git -C "$CODE" rev-parse HEAD)"
echo "branch=$(git -C "$CODE" branch --show-current)"
echo "status_begin"; git -C "$CODE" status --short; echo "status_end"

echo '=== selected resolved contract ==='
"$PY" - "$CONTROL/runtime/resolved.yaml" "$DVAC/runtime/resolved.yaml" <<'PY'
import json, sys, yaml

def get(x, path):
    for part in path.split('.'):
        x = x[part]
    return x

paths = [
    "runner.max_steps", "runner.val_check_interval", "runner.save_interval",
    "env.train.total_num_envs", "env.train.rollout_epoch",
    "env.train.max_episode_steps", "env.train.max_steps_per_rollout_epoch",
    "env.eval.total_num_envs", "env.eval.rollout_epoch",
    "actor.global_batch_size", "actor.micro_batch_size",
    "algorithm.update_epoch", "algorithm.adv_type", "algorithm.loss_type",
    "algorithm.gamma", "algorithm.gae_lambda", "algorithm.logprob_type",
    "actor.model.add_value_head", "actor.model.num_action_chunks",
    "actor.model.num_steps", "actor.fsdp_config.checkpoint_format",
    "algorithm.dvac_gradient_weighting.mode",
    "algorithm.dvac_gradient_weighting.application",
    "algorithm.dvac_gradient_weighting.selected_denoise_last_n",
    "algorithm.dvac_gradient_weighting.history_window_steps",
    "algorithm.dvac_gradient_weighting.warmup_steps",
    "algorithm.dvac_gradient_weighting.weight_min",
    "algorithm.dvac_gradient_weighting.weight_max",
]
for label, path in (("control", sys.argv[1]), ("dvac", sys.argv[2])):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    out = {}
    for key in paths:
        try:
            out[key] = get(cfg, key)
        except (KeyError, TypeError):
            out[key] = "<MISSING>"
    print(label + "=" + json.dumps(out, ensure_ascii=False, sort_keys=True))
PY

echo '=== dvac terminal evidence ==='
echo 'tail:'
tail -n 180 "$DVAC/runtime/driver.log"
echo 'fatal-like lines with line numbers:'
grep -aniE 'Traceback|out of memory|OutOfMemory|CUDA error|NCCL|worker died|WorkerCrashedError|RayActorError|actor died|ErrorInitializationFailed|vk::|fatal|SIGTERM|SIGKILL|terminated|killed|shutdown' "$DVAC/runtime/driver.log" | tail -n 100 || true

echo '=== control matching lines ==='
grep -aniE 'Traceback|out of memory|OutOfMemory|CUDA error|NCCL|worker died|WorkerCrashedError|RayActorError|actor died|ErrorInitializationFailed|vk::|fatal|SIGTERM|SIGKILL|terminated|killed|shutdown' "$CONTROL/runtime/driver.log" | tail -n 40 || true

echo '=== live process/gpu ownership ==='
ps -eo user=,pid=,etimes=,stat=,args= | grep -E 'ppo-control-formal100|ppo-dvac-action-adv-fix-w0p5to1p5' | grep -v grep || true
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
for gpu in 4 5 6 7; do
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    echo "gpu=$gpu pid=$pid user=$(ps -o user= -p "$pid" | xargs) cmd=$(ps -o args= -p "$pid" | cut -c1-180)"
  done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
done
