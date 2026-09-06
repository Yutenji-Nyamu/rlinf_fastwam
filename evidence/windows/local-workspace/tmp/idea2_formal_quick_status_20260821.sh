#!/usr/bin/env bash
set -u
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821

date --iso-8601=seconds
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$run/metrics.log" | tail -n 1
pgrep -af 'train_embodied_agent.py.*robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory_current='; cat /sys/fs/cgroup/memory.current
printf 'memory_peak='; cat /sys/fs/cgroup/memory.peak
printf 'memory_events='; tr '\n' ' ' </sys/fs/cgroup/memory.events; printf '\n'
printf 'latest_checkpoints='; find "$run/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null | sort -V | tail -c 200; printf '\n'
for f in driver.exitcode launch_finished_at.txt; do
  if [[ -f "$runtime/$f" ]]; then printf '%s=' "$f"; cat "$runtime/$f"; else printf '%s=missing\n' "$f"; fi
done
