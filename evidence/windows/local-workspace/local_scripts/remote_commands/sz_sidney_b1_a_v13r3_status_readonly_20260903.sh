set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3
printf '%s\n' '--- time/process ---'
date --iso-8601=seconds
ps -eo pid,ppid,etime,user,args --sort=pid | grep -E 'eval_embodied_agent|pi05_sidney_b1_adjust_badseed_retry_m10_phys4_evalrunner_v13r3|sidney-pi05-current-rlinf' | grep -v grep | tail -n 40 || true
printf '%s\n' '--- gpu4 ---'
nvidia-smi -i 4 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4 --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
printf '%s\n' '--- memory ---'
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
printf '%s\n' '--- runtime files ---'
find "$RUN" -maxdepth 4 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 40 || true
printf '%s\n' '--- driver tail ---'
tail -n 120 "$RUN/runtime/driver.log" 2>/dev/null || true
printf '%s\n' '--- resources tail ---'
tail -n 20 "$RUN/runtime/resources.log" 2>/dev/null || true
printf '%s\n' '--- fatal scan ---'
grep -Ein 'traceback|fatal|oom|out of memory|ray.*error|actor died|exception|unstable|reset|seed|success|reward|episode|action' "$RUN/runtime/driver.log" 2>/dev/null | tail -n 100 || true
