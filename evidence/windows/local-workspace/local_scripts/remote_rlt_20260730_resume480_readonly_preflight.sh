#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
source_runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime
source_run=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1
source_experiment=robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1
checkpoint="${source_run}/${source_experiment}/checkpoints/global_step_250"
completion="${checkpoint}/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"

printf 'NOW\t%s\n' "$(date --iso-8601=seconds)"
printf 'HOST\t%s\n' "$(hostname)"
cd "$repo"
printf 'BRANCH\t%s\n' "$(git branch --show-current)"
printf 'HEAD\t%s\n' "$(git rev-parse HEAD)"
printf 'STATUS_BEGIN\n'
git status --short --branch
printf 'STATUS_END\n'
printf 'UPSTREAM_DELTA\t%s\n' "$(
  git rev-list --left-right --count HEAD...@{upstream}
)"

printf 'SOURCE_EXIT\t%s\n' "$(cat "${source_runtime}/exit_code.txt")"
printf 'SOURCE_FINISHED\t%s\n' "$(cat "${source_runtime}/finished_at.txt")"
printf 'CHECKPOINT\t%s\n' "$checkpoint"
test -d "$checkpoint"
test ! -L "$checkpoint"
test -f "$completion"
printf 'CHECKPOINT_BYTES\t%s\n' "$(du -sb "$checkpoint" | awk '{print $1}')"
printf 'CHECKPOINT_FILES\t%s\n' "$(find "$checkpoint" -type f | wc -l)"
printf 'COMPLETION_SHA256\t%s\n' "$(sha256sum "$completion" | awk '{print $1}')"
printf 'COMPLETION_JSON_BEGIN\n'
python -B - "$completion" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
print(json.dumps(payload, sort_keys=True, indent=2))
assert payload["complete"] is True
assert payload["saved_runner_step"] == 250
assert payload["actor_world_size"] == 2
PY
printf 'COMPLETION_JSON_END\n'

for rank in 0 1; do
  state="${checkpoint}/actor/sac_components/rlt_trainer_state/checkpoint_rank_${rank}.pt"
  replay="${checkpoint}/actor/sac_components/replay_buffer/rank_${rank}"
  test -s "$state"
  test -d "$replay"
  test -f "${replay}/metadata.json"
  printf 'RANK_STATE_%s_BYTES\t%s\n' "$rank" "$(stat -c %s "$state")"
  printf 'RANK_STATE_%s_SHA256\t%s\n' "$rank" "$(
    sha256sum "$state" | awk '{print $1}'
  )"
  printf 'REPLAY_%s_BYTES\t%s\n' "$rank" "$(du -sb "$replay" | awk '{print $1}')"
  printf 'REPLAY_%s_FILES\t%s\n' "$rank" "$(find "$replay" -type f | wc -l)"
  printf 'REPLAY_%s_METADATA\t%s\n' "$rank" "$(
    tr -d '\n' <"${replay}/metadata.json"
  )"
done

printf 'TRAIN_PROCESSES_BEGIN\n'
{
  ps -eo pid=,comm=,args= \
    | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}' || true
  pgrep -ax raylet || true
  pgrep -ax gcs_server || true
}
printf 'TRAIN_PROCESSES_END\n'
printf 'GPU_BEGIN\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'GPU_END\n'
printf 'COMPUTE_PIDS_BEGIN\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
printf 'COMPUTE_PIDS_END\n'
printf 'MEMAVAILABLE_KIB\t%s\n' "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
printf 'CGROUP_CURRENT\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'CGROUP_ANON\t%s\n' "$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'MEMORY_EVENTS\t%s\n' "$(tr '\n' ' ' </sys/fs/cgroup/memory.events)"
printf 'MEMORY_PSI\t%s\n' "$(tr '\n' ' ' </proc/pressure/memory)"
printf 'TMP_AVAIL_BYTES\t%s\n' "$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)"
printf '%s\n' RLT_RESUME480_READONLY_PREFLIGHT_PASS
