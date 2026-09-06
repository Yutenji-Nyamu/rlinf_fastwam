#!/usr/bin/env bash
set -euo pipefail
RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo/runs
CONTROL=$ROOT/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1
DVAC=$ROOT/ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1
date --iso-8601=seconds
for pair in "CONTROL:$CONTROL" "DVAC:$DVAC"; do
  label=${pair%%:*}; run=${pair#*:}; pid=$(cat "$run/runtime/wrapper.pid")
  echo "=== $label ==="
  echo "pid=$pid alive=$([[ -d /proc/$pid ]] && echo 1 || echo 0) exit=$(cat "$run/runtime/exit_code.txt" 2>/dev/null || echo pending)"
  echo "driver_bytes=$(stat -c %s "$run/runtime/driver.log" 2>/dev/null || echo 0)"
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100|Rollout Epoch:[[:space:]]+[0-9]+/4' "$run/runtime/driver.log" 2>/dev/null | tail -n8 || true
  grep -aEi 'traceback|out of memory|CUDA error|worker died|errorinitializationfailed|fatal' "$run/runtime/driver.log" 2>/dev/null | tail -n5 || true
  tail -n8 "$run/runtime/driver.log" 2>/dev/null || true
done
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_ppo_startup_readonly", logging_level="ERROR")
for ns in ("RLinf","RLinf_1"):
    rows=[r for r in ray.util.list_named_actors(all_namespaces=True) if r.get("namespace")==ns]
    print(f"namespace={ns} actors={len(rows)}")
ray.shutdown()
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
for gpu in 4 5 6 7; do
  while read -r pid; do
    [[ -n "$pid" && -r /proc/$pid/environ ]] || continue
    job=$(tr '\0' '\n' < /proc/$pid/environ | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    echo "gpu=$gpu pid=$pid job=$job cmd=$(ps -o comm= -p "$pid")"
  done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
done
awk '/^MemAvailable:|^SwapFree:/' /proc/meminfo
echo '=== RESOLVED CONTRACT ==='
sha256sum "$CONTROL/runtime/resolved.yaml" "$DVAC/runtime/resolved.yaml"
"$VENV/bin/python" - "$CONTROL/runtime/resolved.yaml" "$DVAC/runtime/resolved.yaml" <<'PY'
import json, sys, yaml
c=yaml.safe_load(open(sys.argv[1])); d=yaml.safe_load(open(sys.argv[2]))
def brief(x):
    return {
        "placement":x["cluster"]["component_placement"], "steps":x["runner"]["max_steps"],
        "eval_every":x["runner"]["val_check_interval"], "save_every":x["runner"]["save_interval"],
        "train_env":x["env"]["train"]["total_num_envs"], "rollout":x["env"]["train"]["rollout_epoch"],
        "eval_env":x["env"]["eval"]["total_num_envs"], "gb":x["actor"]["global_batch_size"],
        "mb":x["actor"]["micro_batch_size"], "update":x["algorithm"]["update_epoch"],
        "adv":x["algorithm"]["adv_type"], "loss":x["algorithm"]["loss_type"],
        "group":x["algorithm"]["group_size"], "value_head":x["actor"]["model"]["add_value_head"],
        "logprob":x["algorithm"]["logprob_type"], "dvac":x["algorithm"]["dvac_gradient_weighting"],
        "checkpoint":x["actor"]["fsdp_config"]["checkpoint_format"],
    }
print(json.dumps({"control":brief(c),"dvac":brief(d)}, indent=2))
PY
