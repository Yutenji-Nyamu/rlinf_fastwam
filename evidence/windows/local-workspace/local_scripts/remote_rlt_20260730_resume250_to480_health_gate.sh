#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

venv=/root/autodl-tmp/RLinf/.venv
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
driver="$(cat "$runtime/driver_pid.txt")"
monitor="$(cat "$runtime/monitor_pid.txt")"

kill -0 "$driver"
kill -0 "$monitor"
test ! -e "$runtime/exit_code.txt"
test "$(awk '$1 == "oom" {print $2}' /sys/fs/cgroup/memory.events)" = 0
test "$(awk '$1 == "oom_kill" {print $2}' /sys/fs/cgroup/memory.events)" = 0
grep -Fq 'Resuming training from checkpoint directory' "$runtime/driver.log"
grep -Fq 'Global Step:  251/480' "$runtime/driver.log"

post_resume="$(
  awk '
    /Resuming training from checkpoint directory/ {seen=1}
    seen {print}
  ' "$runtime/driver.log"
)"
if printf '%s\n' "$post_resume" \
  | grep -E \
    'fingerprint mismatch|RLT trainer-state.*failed|CUDA out of memory|NCCL.*(error|fatal)|RayActorError|NaN|Inf|RuntimeError|ValueError|Traceback'
then
  printf '%s\n' "Post-resume fatal signature detected." >&2
  exit 50
fi

event="$(
  find "$run_root/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' \
    | head -n 1
)"
test -n "$event"

"${venv}/bin/python" -B - "$event" <<'PY'
import json
import math
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event = sys.argv[1]
acc = EventAccumulator(event, size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]
wanted_suffixes = (
    "rlt/update_step",
    "rlt/actor_updates_run",
    "rlt/critic_updates_run",
    "rlt/ready_for_online",
    "rlt/pending_update_budget",
    "actor/actor_weight_ramp_progress",
    "actor/bc_weight",
    "actor/q_weight",
    "replay/actor_switch_rate",
    "sac/actor_loss",
    "sac/critic_loss",
    "actor/grad_norm",
    "critic/grad_norm",
    "env/success_once",
    "time/step",
)
payload = {}
for suffix in wanted_suffixes:
    matches = [tag for tag in tags if tag == suffix or tag.endswith("/" + suffix)]
    assert len(matches) == 1, (suffix, matches)
    step_values = [
        value for value in acc.Scalars(matches[0]) if int(value.step) == 250
    ]
    assert len(step_values) == 1, (matches[0], [value.step for value in step_values])
    value = step_values[0]
    assert math.isfinite(value.value), (matches[0], value.value)
    payload[suffix] = {
        "tag": matches[0],
        "step": int(value.step),
        "value": float(value.value),
    }

assert payload["rlt/update_step"]["step"] == 250
assert payload["rlt/update_step"]["value"] == 102260.0
assert payload["rlt/critic_updates_run"]["value"] == 510.0
assert payload["rlt/actor_updates_run"]["value"] == 255.0
assert payload["rlt/ready_for_online"]["value"] == 1.0
assert payload["rlt/pending_update_budget"]["value"] == 0.0
assert payload["actor/actor_weight_ramp_progress"]["value"] == 1.0
assert payload["actor/bc_weight"]["value"] == 2.5
assert math.isclose(
    payload["actor/q_weight"]["value"],
    0.45,
    rel_tol=0.0,
    abs_tol=1e-6,
)
assert payload["replay/actor_switch_rate"]["value"] == 1.0
assert payload["env/success_once"]["value"] == 0.875
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf 'NOW\t%s\n' "$(date --iso-8601=seconds)"
printf 'DRIVER_PID\t%s\n' "$driver"
printf 'MONITOR_PID\t%s\n' "$monitor"
printf 'EVENT\t%s\n' "$event"
printf 'GPU_BEGIN\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'GPU_END\n'
printf 'CGROUP_CURRENT\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'CGROUP_ANON\t%s\n' "$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'MEMORY_EVENTS\t%s\n' "$(tr '\n' ' ' </sys/fs/cgroup/memory.events)"
printf 'RESOURCE_ROWS\t%s\n' "$(wc -l <"$runtime/resources.csv")"
printf '%s\n' RLT_RESUME250_TO480_FIRST_CYCLE_HEALTH_PASS
