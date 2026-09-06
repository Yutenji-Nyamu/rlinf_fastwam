set -euo pipefail

queue_pid="257181"
queue_root="/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue"

echo "=== before ==="
date -Is
if [ -f "$queue_root/pair_launched_at.txt" ]; then
  echo "PAIR_ALREADY_LAUNCHED"
  cat "$queue_root/pair_launched_at.txt"
fi

if kill -0 "$queue_pid" 2>/dev/null; then
  cmdline="$(tr '\0' ' ' < "/proc/$queue_pid/cmdline")"
  pgid="$(ps -o pgid= -p "$queue_pid" | tr -d ' ')"
  echo "queue_pid=$queue_pid queue_pgid=$pgid cmdline=$cmdline"
  case "$cmdline" in
    *queue_after_old.sh*) ;;
    *) echo "REFUSE_UNEXPECTED_QUEUE_CMDLINE"; exit 31 ;;
  esac
  [ "$pgid" = "$queue_pid" ] || { echo "REFUSE_UNEXPECTED_QUEUE_PGID"; exit 32; }
  kill -TERM -- "-$pgid"
  for _ in $(seq 1 20); do
    kill -0 "$queue_pid" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$queue_pid" 2>/dev/null; then
    echo "QUEUE_STILL_ALIVE"
    exit 33
  fi
  echo "QUEUE_CANCELLED"
else
  echo "QUEUE_NOT_ALIVE"
fi

echo "=== scoped process check ==="
ps -eo pid,pgid,stat,lstart,args | grep -E 'rlt_single_gpu_(control|teacher_dvac)|46001|47001' | grep -v grep || true
echo "=== old wrapper check ==="
ps -o pid,pgid,stat,lstart,args -p 106844 || true
