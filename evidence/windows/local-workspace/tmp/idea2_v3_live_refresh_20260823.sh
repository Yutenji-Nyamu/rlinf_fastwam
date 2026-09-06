#!/usr/bin/env bash
set -u
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
for name in wrapper driver observer; do
  pid=$(cat "$runtime/$name.pid" 2>/dev/null || true)
  printf '%s_PID=%s ALIVE=%s\n' "${name^^}" "$pid" "$(test -n "$pid" && test -d "/proc/$pid" && echo 1 || echo 0)"
done
printf 'COMPLETE_STEPS\n'
grep -a 'Global Step:' "$run/metrics.log" 2>/dev/null | tail -n 4 || true
printf 'ACTIVE_ROLLOUT\n'
grep -a 'Generating Rollout Epochs:' "$runtime/driver.log" 2>/dev/null | tail -n 3 || true
printf 'FATAL_MATCHES=%s\n' "$(grep -aEc 'ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL.*(error|Error)|fatal|FATAL' "$runtime/driver.log" 2>/dev/null || true)"
printf 'KNOWN_OPTIONAL_CUROBO_MATCHES=%s\n' "$(grep -ac "No module named 'curobo.types.math'" "$runtime/driver.log" 2>/dev/null || true)"
printf 'CORE_ACTORS\n'
/root/autodl-tmp/RLinf/.venv/bin/ray list actors --detail 2>/dev/null | awk '/class_name: (EmbodiedFSDPActor|MultiStepRolloutWorker|EnvWorker)/ {c=$2} /state:/ && c {print c,$2; c=""}'
printf 'ARTIFACTS\n'
du -sh "$run" "$runtime" 2>/dev/null || true
printf 'NPZ_COUNT=%s CSV_COUNT=%s MP4_COUNT=%s\n' \
  "$(find "$run" -type f -name '*.npz' | wc -l)" \
  "$(find "$run" -type f -name '*.csv' | wc -l)" \
  "$(find "$run" -type f -name '*.mp4' | wc -l)"
find "$run" -maxdepth 2 -type d -name 'global_step_*' -printf '%f\n' | sort -V
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
printf 'RESOURCE_FIRST_LAST\n'
{ head -n 2 "$runtime/resource_monitor/resources.csv"; tail -n 2 "$runtime/resource_monitor/resources.csv"; } 2>/dev/null || true
printf 'MEMORY_CURRENT_BYTES=%s MEMORY_MAX_BYTES=%s\n' "$(cat /sys/fs/cgroup/memory.current)" "$(cat /sys/fs/cgroup/memory.max)"
printf 'MEMORY_EVENTS\n'
grep -E '^max |^oom |^oom_kill ' /sys/fs/cgroup/memory.events
printf 'DISK\n'
df -h /root/autodl-tmp /dev/shm | tail -n +2
printf 'TOP_RSS_KIB\n'
ps -eo pid,rss,comm,args --sort=-rss | awk 'NR<=9 {print}'
printf 'SOURCE\n'
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight rev-parse HEAD
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight status --short
