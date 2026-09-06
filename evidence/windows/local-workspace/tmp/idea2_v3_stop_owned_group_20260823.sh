set -u
pgid=70608
expected='idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822'

echo '=== pre_stop ==='
date '+%Y-%m-%d %H:%M:%S %Z'
members=$(ps -eo pid=,pgid=,args= | awk -v g="$pgid" '$2==g {print}')
printf '%s\n' "$members"
if [ -z "$members" ]; then
  echo 'owned process group already absent'
  exit 0
fi
if ! printf '%s\n' "$members" | grep -q "$expected"; then
  echo 'expected run marker absent; refusing signal'
  exit 21
fi

echo '=== signal ==='
kill -INT -- "-$pgid"
echo "sent SIGINT to PGID $pgid"

echo '=== cleanup_wait ==='
i=0
while [ "$i" -lt 30 ]; do
  if ! ps -eo pgid= | awk -v g="$pgid" '$1==g {found=1} END {exit !found}'; then
    echo "group exited after ${i}s"
    break
  fi
  sleep 1
  i=$((i+1))
done

echo '=== post_stop ==='
date '+%Y-%m-%d %H:%M:%S %Z'
ps -eo pid,ppid,pgid,stat,etime,rss,args | awk -v g="$pgid" 'NR==1 || $3==g'
pgrep -af "$expected" || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
