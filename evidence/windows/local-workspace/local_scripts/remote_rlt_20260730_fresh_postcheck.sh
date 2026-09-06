#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime
run=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1/robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1
ckpt="${run}/checkpoints/global_step_1"
printf 'postcheck_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'exit_code\t%s\n' "$(tr -d '\n' < "${runtime}/exit_code.txt")"
printf 'started_at\t%s\n' "$(tr -d '\n' < "${runtime}/started_at.txt")"
printf 'finished_at\t%s\n' "$(tr -d '\n' < "${runtime}/finished_at.txt")"
printf 'driver_alive\t%s\n' "$(kill -0 "$(cat "${runtime}/driver_pid.txt")" 2>/dev/null && printf yes || printf no)"
printf 'monitor_alive\t%s\n' "$(kill -0 "$(cat "${runtime}/monitor_pid.txt")" 2>/dev/null && printf yes || printf no)"
printf 'run_bytes\t%s\n' "$(du -sb "${run}" | cut -f1)"
printf 'checkpoint_bytes\t%s\n' "$(du -sb "${ckpt}" | cut -f1)"
printf 'checkpoint_files_begin\n'
find "${ckpt}" -type f -printf '%P\t%s\n' | LC_ALL=C sort
printf 'checkpoint_files_end\n'
printf 'small_runtime_files_begin\n'
find "${runtime}" -maxdepth 1 -type f -printf '%f\t%s\n' | LC_ALL=C sort
printf 'small_runtime_files_end\n'
printf 'run_small_files_begin\n'
find "${run}" -type f -size -2M -printf '%P\t%s\n' | LC_ALL=C sort
printf 'run_small_files_end\n'
printf 'metric_lines_begin\n'
python -B - "${runtime}/driver.log" <<'PY'
from pathlib import Path
import re, sys
text = Path(sys.argv[1]).read_text(errors="replace")
# Strip ANSI and carriage returns so the evidence is stable.
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
needles = [
    "Generating Rollout Epochs:", "Evaluating Rollout Epochs:",
    "Saving checkpoint at step 1.", "rlt/actor_updates_run=",
    "rlt/critic_updates_run=", "rlt/global_total_transitions_added=",
    "rlt/updates_to_run=", "replay/actor_switch_rate=",
    "actor/reference_dropout_prob=", "Global Step:",
]
for line in text.splitlines():
    if any(n in line for n in needles):
        print(line.strip())
PY
printf 'metric_lines_end\n'
printf 'json_state_begin\n'
python -B - "${ckpt}" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
for p in sorted(root.rglob("*.json")):
    try:
        obj = json.loads(p.read_text())
    except Exception as exc:
        print(f"{p.relative_to(root)}\tJSON_ERROR\t{type(exc).__name__}")
        continue
    print(f"{p.relative_to(root)}\t{json.dumps(obj, sort_keys=True, separators=(',', ':'))}")
PY
printf 'json_state_end\n'
printf 'fatal_scan_begin\n'
python -B - "${runtime}/driver.log" <<'PY'
from pathlib import Path
import re, sys
text = Path(sys.argv[1]).read_text(errors="replace")
patterns = {
    "cuda_oom": r"CUDA out of memory|OutOfMemoryError",
    "nccl_fatal": r"NCCL[^\n]*(?:error|abort|failed)",
    "nan_metric": r"(?:=|:)\s*nan(?:\s|$)",
    "ray_actor_death": r"(?:RayActorError|actor died|worker died)",
    "segfault": r"segmentation fault|SIGSEGV",
}
for name, pattern in patterns.items():
    print(f"{name}\t{len(re.findall(pattern, text, flags=re.I))}")
PY
printf 'fatal_scan_end\n'
