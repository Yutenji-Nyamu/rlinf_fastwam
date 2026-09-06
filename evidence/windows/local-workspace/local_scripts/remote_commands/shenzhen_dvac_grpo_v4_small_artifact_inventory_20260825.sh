#!/usr/bin/env bash
set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4

date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
echo '=== small_files ==='
for f in "$RUN/runtime/driver.log" "$RUN/runtime/resource.csv" "$RUN/runtime/resolved.yaml"; do
  [[ -f "$f" ]] && stat -c '%s %y %n' "$f"
done
find "$RUN" -type f -name 'events.out.tfevents*' -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -k2,2
echo '=== resource_header_tail ==='
head -n 2 "$RUN/runtime/resource.csv" 2>/dev/null || true
tail -n 3 "$RUN/runtime/resource.csv" 2>/dev/null || true
echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '=== gpu_pid_owner ==='
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  [[ -r "/proc/$p/status" ]] || continue
  used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, -v p="$p" '$1+0==p {gsub(/ /,"",$2); sum+=$2} END{print sum+0}')
  ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,comm=,args= -p "$p" | sed "s/^/gpu_mem_mib=$used /"
done
echo '=== memory_storage ==='
free -h
df -hT / /home /data
df -ih / /home /data
echo '=== run_size ==='
du -sh "$RUN" 2>/dev/null || true
echo SZ_DVAC_GRPO_V4_SMALL_ARTIFACT_INVENTORY_OK
