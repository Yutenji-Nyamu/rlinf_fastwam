set -eu
run=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7
date -Is
tail -n 65 "$run/driver.log"
stat -c '%y %s %n' "$run/driver.log"
nvidia-smi -i 6 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
tail -n 3 "$run/resource.csv"
ps -u chenyiteng -o pid,ppid,etime,pcpu,rss,stat,comm | awk '$2==1025349 || $2==1025351 || $1==1026083 || $1==1026089'
