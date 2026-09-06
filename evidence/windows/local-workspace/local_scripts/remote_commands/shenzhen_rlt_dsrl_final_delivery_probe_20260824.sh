#!/usr/bin/env bash
set -u

rlt_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
rlt_runtime="$rlt_root/runtime"
rlt_event="$rlt_root/tensorboard/events.out.tfevents.1787542941.admin.1389995.0"
dsrl_run=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run
python=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

printf 'time_cst='; TZ=Asia/Shanghai date --iso-8601=seconds
for pair in "rlt:$rlt_runtime/wrapper.pid:$rlt_runtime/exit_code.txt" "dsrl:$dsrl_run/wrapper.pid:$dsrl_run/exit_code.txt"; do
  name=${pair%%:*}; rest=${pair#*:}; pid_file=${rest%%:*}; exit_file=${rest#*:}
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=alive; else state=dead; fi
  exit_code=$(cat "$exit_file" 2>/dev/null || printf missing)
  printf '%s_pid=%s %s_state=%s %s_exit=%s\n' "$name" "${pid:-missing}" "$name" "$state" "$name" "$exit_code"
done

"$python" - "$rlt_event" <<'PY'
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

acc = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
acc.Reload()
tags = (
    "env/success_once",
    "eval/success_once",
    "train/rlt/global_min_replay_size",
    "train/rlt/global_total_transitions_added",
    "train/replay/actor_switch_rate",
    "train/rlt/ready_for_online",
    "train/rlt/update_step",
)
out = {}
available = set(acc.Tags().get("scalars", []))
for tag in tags:
    if tag not in available:
        out[tag] = {"missing": True}
        continue
    values = acc.Scalars(tag)
    out[tag] = {"runner_step": values[-1].step + 1, "value": values[-1].value, "count": len(values)}
print("rlt_latest=" + json.dumps(out, sort_keys=True))
PY

nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -b | sed -n '2p'
