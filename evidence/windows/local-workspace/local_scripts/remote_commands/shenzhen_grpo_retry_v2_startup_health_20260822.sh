#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
DRIVER=$(cat "$RUN/driver.pid")
OBSERVER=$(cat "$RUN/resource_observer.pid")

printf 'timestamp=%s\ndriver=%s\nobserver=%s\n' "$(date --iso-8601=seconds)" "$DRIVER" "$OBSERVER"
kill -0 "$DRIVER"
kill -0 "$OBSERVER"
ps -o pid,ppid,etimes,rss,stat,comm,args -p "$DRIVER","$OBSERVER"

printf '%s\n' 'ray_core_begin'
pgrep -a -u "$(id -u)" -x gcs_server || true
pgrep -a -u "$(id -u)" -x raylet || true
printf '%s\n' 'ray_core_end'

printf '%s\n' 'expected_workers_begin'
ps -u "$(id -u)" -o pid=,ppid=,etimes=,rss=,comm=,args= \
  | grep -E '[r]ay::|[t]rain_embodied_agent.py|[E]nvWorker|[R]olloutWorker|[A]ctorWorker' \
  | sed -n '1,100p' || true
printf '%s\n' 'expected_workers_end'

printf '%s\n' 'driver_progress_begin'
grep -E 'Generating Rollout Epochs|Global Step|Loaded normalization|FSDP|actor|rollout|EnvWorker|Traceback|CUDA out of memory|WorkerCrashed|NCCL|Failed to connect to GCS' \
  "$RUN/driver.log" | tail -n 120 || true
printf '%s\n' 'driver_progress_end'

fatal_count=$(grep -Eic 'Traceback|CUDA out of memory|WorkerCrashed|NCCL.*(error|fatal)|Failed to connect to GCS|GCS may have been killed' "$RUN/driver.log" || true)
printf 'fatal_count=%s\n' "$fatal_count"

printf '%s\n' 'gpu_summary_begin'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' 'gpu_summary_end'
printf '%s\n' 'gpu_compute_4_7_begin'
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
printf '%s\n' 'gpu_compute_4_7_end'

rel=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
printf 'host_mem_available_kib=%s\ncgroup_memory_current_bytes=%s\n' \
  "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" \
  "$(cat "/sys/fs/cgroup$rel/memory.current")"
printf '%s\n' 'cgroup_memory_events_begin'
cat "/sys/fs/cgroup$rel/memory.events"
printf '%s\n' 'cgroup_memory_events_end'
printf '%s\n' 'resource_tail_begin'
tail -n 5 "$RUN/resource.csv" || true
printf '%s\n' 'resource_tail_end'

printf '%s\n' 'fastwam_gpu3_begin'
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
printf '%s\n' 'fastwam_gpu3_end'
printf '%s\n' 'SZ_GRPO_RETRY_V2_STARTUP_HEALTH_DONE'
