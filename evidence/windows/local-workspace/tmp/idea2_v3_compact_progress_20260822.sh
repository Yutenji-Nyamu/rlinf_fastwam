#!/usr/bin/env bash
set -u
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
for name in wrapper driver observer; do
  pid=$(cat "$runtime/$name.pid" 2>/dev/null || true)
  printf '%s_PID=%s ALIVE=%s\n' "${name^^}" "$pid" "$(test -n "$pid" && test -d "/proc/$pid" && echo 1 || echo 0)"
done
stat -c 'DRIVER_LOG_SIZE=%s MTIME=%y' "$runtime/driver.log" 2>/dev/null || true
printf 'PROGRESS\n'
grep -aE 'Generating Rollout Epochs|Global Step|global_step|Traceback|ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|fatal|ERROR' "$runtime/driver.log" 2>/dev/null | tail -n 15 || true
printf 'ACTORS\n'
/root/autodl-tmp/RLinf/.venv/bin/ray list actors --detail 2>/dev/null | awk '/class_name: (EmbodiedFSDPActor|MultiStepRolloutWorker|EnvWorker)/ {c=$2} /state:/ && c {print c,$2; c=""}'
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits
printf 'RESOURCE_LAST\n'; tail -n 2 "$runtime/resource_monitor/resources.csv" 2>/dev/null || true
printf 'MEMORY_CURRENT_BYTES=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'OOM_EVENTS\n'; grep -E '^oom|^oom_kill' /sys/fs/cgroup/memory.events
