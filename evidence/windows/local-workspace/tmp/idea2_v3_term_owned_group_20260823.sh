set -u
pgid=70608
echo '=== pre_term_count ==='
ps -eo pgid= | awk -v g="$pgid" '$1==g {n++} END {print n+0}'
kill -TERM -- "-$pgid"
echo "sent SIGTERM to PGID $pgid"
i=0
while [ "$i" -lt 30 ]; do
  n=$(ps -eo pgid= | awk -v g="$pgid" '$1==g {n++} END {print n+0}')
  if [ "$n" -eq 0 ]; then
    echo "group exited after ${i}s"
    break
  fi
  sleep 1
  i=$((i+1))
done
echo '=== post_term ==='
ps -eo pid,ppid,pgid,stat,args | awk -v g="$pgid" 'NR==1 || $3==g'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
