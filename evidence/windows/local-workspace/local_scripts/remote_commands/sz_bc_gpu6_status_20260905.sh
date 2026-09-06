#!/usr/bin/env bash
set -u
run=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v4
date -Is
for f in wrapper.pid timeout.pid started_at.txt exit_code.txt finished_at.txt; do if test -f "$run/$f"; then printf '%s=' "$f"; cat "$run/$f"; fi; done
tail -n 70 "$run/driver.log" 2>/dev/null
tail -n 5 "$run/resource.csv" 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 6 --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits
free -h
df -h /data
ps -o user:16,pid,ppid,etime,pcpu,rss,args -p 3176215,321933,322685 | cut -c 1-550
find "$run" -maxdepth 6 -type f \( -name '*shard*' -o -name '*.distcp' -o -name '.metadata' -o -name '*replay.pt' -o -name 'learner.pt' \) -printf '%p %s bytes\n'
