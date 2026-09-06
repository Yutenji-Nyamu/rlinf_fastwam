#!/usr/bin/env bash
set -u

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

printf 'IDENTITY\n'
hostname
pwd
id -u
date --iso-8601=seconds

printf 'ROOT_PIDS\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime_dir/${name}.pid" 2>/dev/null || true)
  if [[ -n "$pid" && -d "/proc/$pid" ]]; then
    stat=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
    rss_kib=$(awk '/VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
    printf '%s\t%s\talive\tstate=%s\trss_kib=%s\n' "$name" "$pid" "$stat" "$rss_kib"
  else
    printf '%s\t%s\tdead_or_missing\n' "$name" "${pid:-missing}"
  fi
done

printf 'PROCESS_TREE\n'
ps -eo pid,ppid,stat,rss,etime,cmd --sort=-rss | grep -E 'idea2_dvac_r_only_downweight_formal|ray::|raylet|gcs_server' | grep -v grep | head -n 80 || true

printf 'LATEST_GLOBAL_STEP_LINES\n'
grep -a 'Global Step' "$run_dir/metrics.log" 2>/dev/null | tail -n 12 || true

printf 'LATEST_TRAIN_METRICS\n'
grep -aE 'rollout/.*/success_once|rollout/.*/episode_reward|actor/.*approx_kl|actor/.*clip_fraction|actor/.*grad_norm|policy_loss|Global Step' "$run_dir/metrics.log" 2>/dev/null | tail -n 120 || true

printf 'RUNNER_CSV_TAILS\n'
for csv in "$run_dir"/dvac_train/actor_rank*/runner_step_metrics.csv; do
  printf 'FILE=%s\n' "$csv"
  head -n 1 "$csv"
  tail -n 4 "$csv"
done

printf 'LATEST_COMMON_NPZ\n'
for rank in actor_rank00 actor_rank01; do
  find "$run_dir/dvac_train/$rank" -maxdepth 1 -type f -name 'rollout_step*.npz' -printf '%f\t%s\n' 2>/dev/null | sort -V | tail -n 5
done

printf 'CHECKPOINTS\n'
for d in "$run_dir"/checkpoints/global_step_*; do
  [[ -d "$d" ]] || continue
  du -sh "$d"
  find "$d" -type f | wc -l | awk -v d="$d" '{print d "\tfiles=" $1}'
done

printf 'ARTIFACT_COUNTS_AND_SIZES\n'
du -sh "$run_dir" "$runtime_dir" "$run_dir/dvac_train" "$run_dir/checkpoints" "$run_dir/control_trace" 2>/dev/null || true
printf 'dvac_npz_count='; find "$run_dir/dvac_train" -type f -name 'rollout_step*.npz' 2>/dev/null | wc -l
printf 'control_trace_files='; find "$run_dir/control_trace" -type f 2>/dev/null | wc -l
printf 'control_trace_mp4='; find "$run_dir/control_trace" -type f -name '*.mp4' 2>/dev/null | wc -l
printf 'control_trace_frames_rows='; find "$run_dir/control_trace" -type f -name 'frames.csv' -exec awk 'FNR>1{n++} END{print n+0}' {} + 2>/dev/null || true
find "$run_dir/control_trace" -type f -printf '%p\t%s\n' 2>/dev/null | sort || true

printf 'GPU_CURRENT\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits

printf 'CGROUP_AND_RAM\n'
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.peak='; cat /sys/fs/cgroup/memory.peak 2>/dev/null || true
cat /sys/fs/cgroup/memory.events
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo

printf 'RESOURCE_CSV_HEAD_TAIL\n'
head -n 2 "$runtime_dir/resource_monitor/resources.csv" 2>/dev/null || true
tail -n 5 "$runtime_dir/resource_monitor/resources.csv" 2>/dev/null || true

printf 'DISK\n'
df -h /root/autodl-tmp
df -i /root/autodl-tmp

printf 'LOG_AND_RUNTIME_FILES\n'
stat -c '%n\t%s\t%y' "$run_dir/metrics.log" "$runtime_dir/driver.log" "$runtime_dir/resource_monitor/resources.csv" "$runtime_dir/resource_monitor/process_rss.tsv" 2>/dev/null || true
find "$runtime_dir" -maxdepth 2 -type f -printf '%p\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' 2>/dev/null | sort

printf 'FATAL_SCAN\n'
patterns='CUDA out of memory|OutOfMemoryError|NCCL.*(error|Error)|WorkerCrashed|RayTaskError|No space left|SIGSEGV|SIGBUS|Killed|Traceback|fatal|Fatal|FATAL'
for f in "$run_dir/metrics.log" "$runtime_dir/driver.log" "$runtime_dir/observer.log"; do
  [[ -f "$f" ]] || continue
  count=$(grep -aEc "$patterns" "$f" 2>/dev/null || true)
  printf '%s\tcount=%s\n' "$f" "$count"
  grep -aE "$patterns" "$f" 2>/dev/null | tail -n 12 || true
done

printf 'COMPLETION_MARKERS\n'
for f in "$runtime_dir"/*rc "$runtime_dir"/*exit* "$runtime_dir"/*done*; do
  [[ -e "$f" ]] && stat -c '%n\t%s\t%y' "$f"
done
