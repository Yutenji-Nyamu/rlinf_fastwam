#!/usr/bin/env bash
set -u

run=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run
log="$run/driver.log"
ckpt="$run/dsrl-current-formal-200c-v2/checkpoints"
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389

printf '%s\n' '--- sentinel time ---'
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds

printf '%s\n' '--- DSRL owner chain ---'
wrapper=$(cat "$run/wrapper.pid" 2>/dev/null || true)
printf 'wrapper_pid=%s\n' "${wrapper:-missing}"
if [ -n "$wrapper" ]; then
  ps --forest -o pid=,ppid=,pgid=,etimes=,stat=,rss=,cmd= -g "$wrapper" 2>/dev/null || true
fi

printf '%s\n' '--- DSRL Ray actors (job 08000000) ---'
timeout 20s env RAY_ADDRESS="$ray_address" "$venv/bin/ray" list actors \
  --filter 'job_id=08000000' --detail 2>&1 \
  | grep -E 'actor_id:|class_name:|state:|job_id:|name:|pid:|ray_namespace:|No resource' || true

printf '%s\n' '--- DSRL complete metrics ---'
event=$(find "$run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents*' 2>/dev/null | head -n 1)
if [ -n "$event" ]; then
  "$venv/bin/python" - "$event" <<'PY'
import json
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()

def last(tag):
    if tag not in ea.Tags().get("scalars", []):
        return None
    xs = ea.Scalars(tag)
    if not xs:
        return None
    x = xs[-1]
    return {"step": int(x.step) + 1, "value": float(x.value), "wall_time": float(x.wall_time)}

tags = [
    "env/success_once", "eval/success_once",
    "train/sac/global_resident_transitions", "train/sac/global_new_transitions",
    "train/sac/planned_optimizer_updates", "train/sac/update_step",
    "train/sac/actor_loss", "train/sac/critic_loss", "train/sac/alpha",
    "train/actor/entropy", "train/actor/grad_norm", "train/critic/grad_norm",
    "train/actor/q_pi", "train/critic/q_data", "time/step",
]
out = {tag: last(tag) for tag in tags}
out["all_last_values_finite"] = all(
    item is None or math.isfinite(item["value"]) for item in out.values()
)
print(json.dumps(out, sort_keys=True))
PY
fi
printf 'driver_complete_step_tables='; tr '\r' '\n' < "$log" 2>/dev/null | grep -c 'Global Step:' || true
printf '%s\n' 'driver_progress_tail='
tr '\r' '\n' < "$log" 2>/dev/null \
  | grep -E 'Global Step:|Saving checkpoint|resident_transitions=|planned_optimizer_updates=|success_once=' \
  | tail -n 24 || true

printf '%s\n' '--- DSRL errors and markers ---'
for pattern in 'Traceback' 'CUDA out of memory' 'OutOfMemory' 'WorkerCrashed' 'RayTaskError' 'nonfinite' 'NCCL'; do
  count=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -i -c "$pattern" || true)
  printf '%s=%s\n' "$pattern" "$count"
done
if [ -s "$run/exit_code.txt" ]; then
  printf 'exit_code='; tr '\n' ' ' < "$run/exit_code.txt"; printf '\n'
else
  printf 'exit_code=none\n'
fi
stat -c 'driver_log_bytes=%s mtime=%y' "$log" 2>/dev/null || true

printf '%s\n' '--- DSRL live resources ---'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits
printf '%s\n' 'gpu_processes='
nvidia-smi -i 6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
free -b | sed -n '1,2p'
df -B1 /data | tail -n 1
printf '%s\n' 'resource_tail='
tail -n 2 "$run/resource.csv" 2>/dev/null || true

printf '%s\n' '--- DSRL checkpoints ---'
find "$ckpt" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f|mtime=%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort -V
for d in "$ckpt"/global_step_*; do
  [ -d "$d" ] && du -sb "$d" 2>/dev/null || true
done

printf '%s\n' '--- shared Ray head ---'
ps -eo pid=,ppid=,etimes=,stat=,rss=,cmd= \
  | grep -E '[g]cs_server.*--gcs-server-port=6389|[r]aylet.*--gcs-address=172.17.0.1:6389' || true
