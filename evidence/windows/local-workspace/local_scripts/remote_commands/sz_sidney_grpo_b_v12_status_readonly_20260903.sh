set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12
printf '%s\n' '--- time/process ---'
date --iso-8601=seconds
ps -eo pid,ppid,etime,user,args --sort=pid | grep -E 'train_embodied_agent|move-grpo-smoke1-2gpu64x1|pi05_sidney_move_grpo_smoke1' | grep -v grep | tail -n 60 || true
printf '%s\n' '--- gpu45 ---'
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4,5 --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader || true
printf '%s\n' '--- memory ---'
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
printf '%s\n' '--- runtime files ---'
find "$RUN" -maxdepth 6 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 80 || true
printf '%s\n' '--- driver tail ---'
tail -n 180 "$RUN/runtime/driver.log" 2>/dev/null || true
printf '%s\n' '--- resources tail ---'
tail -n 30 "$RUN/runtime/resources.log" 2>/dev/null || true
printf '%s\n' '--- key lines ---'
grep -Ein 'global step|train/|eval/|success|reward|filter|filtered|kl|clip|grad|loss|optimizer|record|saving|saved|checkpoint|traceback|fatal|oom|out of memory|actor died|exception' "$RUN/runtime/driver.log" 2>/dev/null | tail -n 180 || true
