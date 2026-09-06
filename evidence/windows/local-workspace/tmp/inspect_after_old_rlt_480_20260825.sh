set -euo pipefail

queue_root="/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue"
old_runtime="/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime"
old_run="/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1"
archive_root="/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/high_info_archive_20260825"

echo "=== identity/time ==="
hostname
date -Is

echo "=== queue files ==="
find "$queue_root" -maxdepth 2 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort || true
for f in "$queue_root"/queue.log "$queue_root"/queue_v1.log; do
  if [ -f "$f" ]; then
    echo "--- $f tail ---"
    tail -n 120 "$f"
  fi
done

echo "=== old lifecycle ==="
find "$old_runtime" -maxdepth 2 -type f \( -name '*exit*' -o -name '*end*' -o -name '*lifecycle*' -o -name '*status*' \) -printf '%s %p\n' 2>/dev/null | sort || true
for f in "$old_runtime"/exit_code.txt "$old_runtime"/end_time.txt "$old_runtime"/lifecycle.log; do
  [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }
done

echo "=== old metrics tail ==="
tail -n 30 "$old_run/metrics.log" 2>/dev/null || true

echo "=== archive files ==="
find "$archive_root" -maxdepth 2 -type f -printf '%s %p\n' 2>/dev/null | sort || true
if [ -f "$archive_root/rlt_teacher_dvac_w0to2_formal_fresh480_high_info_raw_20260825.tar.gz" ]; then
  sha256sum "$archive_root/rlt_teacher_dvac_w0to2_formal_fresh480_high_info_raw_20260825.tar.gz"
fi

echo "=== scoped processes ==="
ps -eo pid,pgid,stat,lstart,args | grep -E 'rlt_single_gpu_(control|teacher_dvac)|46001|47001|rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1' | grep -v grep || true

echo "=== resources ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.events='; tr '\n' ' ' < /sys/fs/cgroup/memory.events; echo
