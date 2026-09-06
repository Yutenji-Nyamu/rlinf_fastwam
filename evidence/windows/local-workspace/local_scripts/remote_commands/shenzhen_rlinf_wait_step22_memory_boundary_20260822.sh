#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
LOG="$RUN/driver.log"
PID=1375834
deadline=$(( $(date +%s) + 900 ))

while :; do
  latest=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$LOG" 2>/dev/null | tail -n 1 | grep -oE '[0-9]+' | head -n 1 || true)
  now=$(date --iso-8601=seconds)
  echo "poll_time=$now latest_complete_step=${latest:-none}"
  if [ "${latest:-0}" -ge 22 ]; then
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo 'boundary_timeout=1'
    exit 2
  fi
  sleep 20
done

echo '=== BOUNDARY_TIME ==='
date --iso-8601=seconds
echo '=== MEM_KIB ==='
awk '/^(MemAvailable|AnonPages|Cached|Shmem):/ {print $1,$2}' /proc/meminfo
echo '=== CGROUP ==='
cg=$(awk -F: '$1=="0" {print $3}' "/proc/$PID/cgroup")
printf 'cgroup_path=%s\n' "$cg"
printf 'memory_current_bytes='; cat "/sys/fs/cgroup$cg/memory.current"
echo 'memory_events:'
cat "/sys/fs/cgroup$cg/memory.events"
echo '=== ENVWORKER_RSS_KIB ==='
ps -u "$(id -u)" -o pid=,rss=,etime=,comm=,cmd= | awk '$4=="ray::EnvWorker" {print $1,$2,$3,$4,$5}'
echo '=== USER_RSS_KIB ==='
ps -u "$(id -u)" -o rss= | awk '{s+=$1} END{print s}'
echo '=== GPU ==='
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo '=== STEP22_METRIC ==='
grep -aE 'Global Step:[[:space:]]+22/100|Step Time:|success_once=|actor/approx_kl=|actor/clip_fraction=|actor/grad_norm=|critic/explained_variance=|critic/value_loss=' "$LOG" | tail -n 18
echo '=== FATAL_COUNTS ==='
printf 'traceback='; grep -ac 'Traceback' "$LOG" || true
printf 'cuda_oom='; grep -aci 'CUDA out of memory' "$LOG" || true
printf 'nccl_error='; grep -aciE 'NCCL.*(error|fail|timeout)' "$LOG" || true
printf 'worker_died='; grep -aciE 'worker.*(died|crash)|RayActorError|WorkerCrashedError' "$LOG" || true
printf 'nan_inf='; grep -aciE '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" || true
echo 'FORMAL100_STEP22_BOUNDARY_DONE'
