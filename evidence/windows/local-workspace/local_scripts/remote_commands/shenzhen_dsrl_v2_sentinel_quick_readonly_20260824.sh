#!/usr/bin/env bash
set -u

run=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run
log="$run/driver.log"
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389

TZ=Asia/Shanghai date --iso-8601=seconds
wrapper=$(cat "$run/wrapper.pid" 2>/dev/null || true)
printf 'owner_chain\n'
[ -n "$wrapper" ] && ps --forest -o pid=,ppid=,pgid=,etimes=,stat=,rss=,cmd= -g "$wrapper" 2>/dev/null || true

printf 'main_actors\n'
timeout 20s env RAY_ADDRESS="$ray_address" "$venv/bin/ray" list actors \
  --filter 'job_id=08000000' --detail 2>&1 \
  | awk '
      /class_name: (EmbodiedSACFSDPPolicy|MultiStepRolloutWorker|EnvWorker)/ {keep=1; block=$0 ORS; next}
      keep {block=block $0 ORS}
      keep && /ray_namespace:/ {printf "%s", block; keep=0; block=""}
    ' || true

event=$(find "$run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents*' 2>/dev/null | head -n 1)
TF_CPP_MIN_LOG_LEVEL=3 "$venv/bin/python" - "$event" <<'PY'
import json, math, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); ea.Reload()
tags = ea.Tags().get("scalars", [])
def last(tag):
    if tag not in tags: return None
    x = ea.Scalars(tag)[-1]
    return [int(x.step)+1, float(x.value), float(x.wall_time)]
wanted = [
    "env/success_once", "eval/success_once", "train/sac/global_resident_transitions",
    "train/sac/planned_optimizer_updates", "train/sac/actor_loss", "train/sac/critic_loss",
    "train/sac/alpha", "train/actor/entropy", "train/actor/grad_norm",
    "train/critic/grad_norm", "time/step",
]
out = {tag: last(tag) for tag in wanted}
out["finite"] = all(v is None or math.isfinite(v[1]) for v in out.values())
print(json.dumps(out, sort_keys=True))
PY

printf 'driver_tables='; tr '\r' '\n' < "$log" 2>/dev/null | grep -c 'Global Step:' || true
for pat in 'Traceback' 'CUDA out of memory' 'OutOfMemory' 'WorkerCrashed' 'RayTaskError' 'nonfinite' 'NCCL'; do
  printf '%s=' "$pat"; tr '\r' '\n' < "$log" 2>/dev/null | grep -i -c "$pat" || true
done
[ -s "$run/exit_code.txt" ] && { printf 'exit_code='; tr '\n' ' ' < "$run/exit_code.txt"; printf '\n'; } || printf 'exit_code=none\n'
stat -c 'driver_bytes=%s mtime=%y' "$log" 2>/dev/null || true

printf 'gpu\n'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits
printf 'resource_last\n'; tail -n 1 "$run/resource.csv" 2>/dev/null || true
printf 'checkpoints\n'; find "$run/dsrl-current-formal-200c-v2/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f|%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort -V
