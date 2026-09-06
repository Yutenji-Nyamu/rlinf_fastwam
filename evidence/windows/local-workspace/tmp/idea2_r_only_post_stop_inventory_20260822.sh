#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

printf 'IDENTITY_TIME\n'
hostname
pwd
id -u
date --iso-8601=seconds

printf 'CONTROLLERS_AND_TARGET_PROCESSES\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime/${name}.pid" 2>/dev/null || true)
  if [[ -n "$pid" && -d "/proc/$pid" ]]; then
    printf '%s\t%s\talive\t%s\n' "$name" "$pid" "$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  else
    printf '%s\t%s\texited\n' "$name" "${pid:-missing}"
  fi
done
pgrep -af 'idea2_dvac_r_only_downweight_formal_100step|RLinf_idea2_dvac_residual_downweight.*train_embodied_agent' || true

printf 'FINAL_STEPS_AND_METRICS\n'
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$run/metrics.log" 2>/dev/null | tail -n 10 || true
grep -aE 'Global Step|rollout/.*/success_once|actor/.*approx_kl|actor/.*clip_fraction|actor/.*grad_norm|actor/dvac_weight_mean_rank_local' "$run/metrics.log" 2>/dev/null | tail -n 80 || true

printf 'CHECKPOINTS\n'
find "$run" -maxdepth 7 -type d -name 'global_step_*' -printf '%p\n' 2>/dev/null | sort -V || true
last_ckpt=$(find "$run" -maxdepth 7 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1)
if [[ -n "$last_ckpt" ]]; then
  du -sb "$last_ckpt"
  find "$last_ckpt" -type f -printf '%f\t%s\n' | sort
  printf 'partial_count='; find "$last_ckpt" -type f -name '*.partial' | wc -l
fi

printf 'ARTIFACT_INVENTORY\n'
du -sh "$run" "$runtime" "$run/dvac_train" "$run/control_trace" 2>/dev/null || true
printf 'npz_count='; find "$run/dvac_train" -type f -name 'rollout_step*.npz' | wc -l
printf 'rank_csv_count='; find "$run/dvac_train" -type f -name 'runner_step_metrics.csv' | wc -l
printf 'control_trace_files='; find "$run/control_trace" -type f | wc -l
find "$run/control_trace" -type f -printf '%p\t%s\n' 2>/dev/null | sort || true

printf 'RESOURCE_FINAL\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader || true
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
tail -n 4 "$runtime/resource_monitor/resources.csv" 2>/dev/null || true
df -h /root/autodl-tmp

printf 'EXIT_AND_FILES\n'
for f in driver.exitcode observer.exitcode launch_started_at.txt launch_finished_at.txt resolved_config.yaml launch_command.txt; do
  if [[ -f "$runtime/$f" ]]; then
    stat -c '%n\t%s\t%y' "$runtime/$f"
    case "$f" in *.exitcode|launch_*at.txt) printf 'value='; cat "$runtime/$f";; esac
  fi
done
stat -c '%n\t%s\t%y' "$run/metrics.log" "$runtime/driver.log" "$runtime/resource_monitor/resources.csv" 2>/dev/null || true

printf 'ERROR_SCAN\n'
patterns='CUDA out of memory|OutOfMemoryError|NCCL.*(error|Error)|WorkerCrashed|RayTaskError|No space left|SIGSEGV|SIGBUS|Killed|Traceback|fatal|Fatal|FATAL'
for f in "$run/metrics.log" "$runtime/driver.log" "$runtime/observer.log"; do
  [[ -f "$f" ]] || continue
  count=$(grep -aEc "$patterns" "$f" 2>/dev/null || true)
  printf '%s\tcount=%s\n' "$f" "$count"
  grep -aE "$patterns" "$f" 2>/dev/null | tail -n 20 || true
done
