#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
RLT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
DSRL=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
UID_NOW=$(id -u)
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)

printf '=== IDENTITY_TIME ===\n'
date --iso-8601=seconds
hostname
id

printf '=== GRPO_PROCESS_PHASE ===\n'
printf 'driver_pid=%s\n' "$DRIVER"
if test -n "$DRIVER" && kill -0 "$DRIVER" 2>/dev/null; then
  ps -o user=,pid=,ppid=,pgid=,etimes=,rss=,stat=,comm=,args= -p "$DRIVER"
else
  printf 'driver_alive=0\n'
fi
printf 'actor='; pgrep -u "$UID_NOW" -fc '^ray::EmbodiedFSDPActor' || true
printf 'rollout='; pgrep -u "$UID_NOW" -fc '^ray::MultiStepRolloutWorker' || true
printf 'env='; pgrep -u "$UID_NOW" -fc '^ray::EnvWorker' || true
printf 'raylet='; pgrep -u "$UID_NOW" -xc raylet || true
printf 'gcs='; pgrep -u "$UID_NOW" -xc gcs_server || true
printf '%s\n' '-- progress tail --'
grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$RUN/driver.log" 2>/dev/null | tail -n 36 || true
printf '%s\n' '-- latest complete metric block --'
tail -n 180 "$RUN/metrics.log" 2>/dev/null || true
printf '%s\n' '-- latest fixed eval excerpt --'
grep -a -n -E 'Global Step:|Evaluation|success_at_end=|success_once=' "$RUN/driver.log" 2>/dev/null | tail -n 30 || true
printf '%s\n' '-- exits/fatals --'
for file in driver.exit resource_observer.exit; do
  if test -f "$RUN/$file"; then printf '%s=' "$file"; cat "$RUN/$file"; else printf '%s=missing\n' "$file"; fi
done
for file in "$RUN/metrics.log" "$RUN/driver.log" "$RUN/resource_observer.log"; do
  test -f "$file" || continue
  fatal=$(grep -aEic 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|Failed to connect to GCS|GCS may have been killed|No space left' "$file" || true)
  printf '%s fatal_pattern_count=%s\n' "$file" "$fatal"
done
printf 'metrics_nonfinite='; grep -aEic '(^|[^A-Za-z])(nan|inf|-inf)([^A-Za-z]|$)' "$RUN/metrics.log" 2>/dev/null || true
printf '%s\n' '-- checkpoints --'
find "$RUN" -maxdepth 2 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 12

printf '=== GPU_ALL ===\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits
printf '%s\n' '-- compute apps --'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
printf '%s\n' '-- mapped gpu pids --'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u); do
  test -n "$pid" && ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm=,args= -p "$pid" 2>/dev/null || true
done

printf '=== HOST_MEMORY ===\n'
free -h
awk '/^(MemAvailable|SwapTotal|SwapFree):/ {print}' /proc/meminfo
printf 'memory_psi\n'; cat /proc/pressure/memory 2>/dev/null || true
printf 'vmstat\n'; vmstat 1 2
if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  cg="/sys/fs/cgroup$rel"
  printf 'cgroup=%s\n' "$cg"
  for name in memory.current memory.peak memory.swap.current memory.swap.peak; do
    test -r "$cg/$name" && printf '%s=%s\n' "$name" "$(cat "$cg/$name")"
  done
  printf 'memory.events\n'; cat "$cg/memory.events" 2>/dev/null || true
  printf 'memory.pressure\n'; cat "$cg/memory.pressure" 2>/dev/null || true
fi

printf '=== DISKS ===\n'
df -h / /home /data
df -i / /home /data

printf '=== GIT_WORKTREES ===\n'
for item in "RLT:$RLT" "DSRL:$DSRL"; do
  name=${item%%:*}; path=${item#*:}
  printf '%s path=%s\n' "$name" "$path"
  if test -d "$path/.git" || git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'head='; git -C "$path" rev-parse HEAD
    printf 'branch='; git -C "$path" branch --show-current
    printf 'status='; test -z "$(git -C "$path" status --porcelain=v1)" && echo clean || { echo dirty; git -C "$path" status --short; }
    printf 'upstream='; git -C "$path" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || echo NONE
    printf 'ahead_behind='; git -C "$path" rev-list --left-right --count '@{upstream}...HEAD' 2>/dev/null || echo UNKNOWN
    git -C "$path" remote -v | sort -u
  else
    echo missing
  fi
done

printf '=== NETWORK_LIGHT ===\n'
printf 'proxy_env='; env | grep -iE '^(http|https|all|no)_proxy=' | sort | tr '\n' ' '; echo
printf 'mihomo_service='; systemctl is-active mihomo 2>/dev/null || true
ss -ltn 2>/dev/null | grep -E '127\.0\.0\.1:(7890|9090)' || true
probe() {
  label=$1; shift
  printf '%s ' "$label"
  "$@" -o /dev/null -sS -L --max-time 8 -w 'code=%{http_code} connect=%{time_connect} total=%{time_total} remote=%{remote_ip}\n' || printf 'curl_exit=%s\n' "$?"
}
probe direct_github env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl https://github.com/
probe direct_hf env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl 'https://huggingface.co/api/models?limit=1'
probe proxy_github curl -x http://127.0.0.1:7890 https://github.com/
probe proxy_hf curl -x http://127.0.0.1:7890 'https://huggingface.co/api/models?limit=1'

printf 'SZ_FINAL_USER_REFRESH_OK\n'
