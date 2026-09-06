set -euo pipefail
ps -eo user,pid,ppid,etime,stat,wchan:24,args --forest | grep -A8 -B2 '1392777' | grep -v grep || true
for p in 1392777 1392845; do
  echo "=== children $p ==="
  ps --ppid "$p" -o user=,pid=,ppid=,stat=,wchan:24,args= || true
done
