#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
source_config="$repo/examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml"
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime

printf 'CHECKED_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'HOSTNAME\t%s\n' "$(hostname)"
printf 'PWD\t%s\n' "$PWD"
printf 'UID\t%s\n' "$(id -u)"

printf '%s\n' '=== target git/source ==='
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short --branch
git -C "$repo" rev-list --left-right --count HEAD...@{upstream}
sha256sum "$source_config" "$norm_stats"

printf '%s\n' '=== collision paths ==='
for path in "$run_root" "$runtime_root"; do
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'EXISTS\t%s\n' "$path"
  else
    printf 'ABSENT\t%s\n' "$path"
  fi
done

printf '%s\n' '=== active training/ray ==='
ps -eo pid=,comm=,args= \
  | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}' || true
pgrep -ax raylet || true
pgrep -ax gcs_server || true

printf '%s\n' '=== gpu ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits
printf '%s\n' '=== compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits || true

printf '%s\n' '=== memory/cgroup ==='
free -b
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events

printf '%s\n' '=== disk ==='
df -B1 /root/autodl-tmp

printf '%s\n' '=== prior formal completion ==='
prior=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
printf 'PRIOR_EXIT\t%s\n' "$(cat "$prior/exit_code.txt" 2>/dev/null || printf MISSING)"
printf 'PRIOR_FINISHED\t%s\n' "$(cat "$prior/finished_at.txt" 2>/dev/null || printf MISSING)"

printf '%s\n' OGPO_FORMAL_V2_READONLY_PREFLIGHT_COMPLETE
