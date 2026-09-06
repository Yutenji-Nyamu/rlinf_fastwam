set -euo pipefail
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-pi0style-offload-smoke1-2gpu32x8-g8-b2048-u2-m10-phys67-v1
R="$RUN/runtime"
echo "NOW=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
echo '===ERROR_CONTEXT===' 
grep -aBn -m1 'torch.OutOfMemoryError' "$R/driver.log" || true
line=$(grep -aBn -m1 'torch.OutOfMemoryError' "$R/driver.log" | cut -d: -f1 || true)
if test -n "$line"; then
  from=$((line>45 ? line-45 : 1)); to=$((line+25)); sed -n "${from},${to}p" "$R/driver.log"
fi
echo '===RESOURCE_PEAK===' 
awk -F, 'NR==1{next} {if($3>g6){g6=$3;t6=$1};if($5>g7){g7=$5;t7=$1};if(min==0||$2<min){min=$2;tm=$1};n++} END{printf "samples=%d gpu6_peak_mib=%d at=%s gpu7_peak_mib=%d at=%s mem_available_min_kib=%d at=%s\n",n,g6,t6,g7,t7,min,tm}' "$R/resource_2s.csv"
echo '===OWNED_PROCESSES===' 
ps -u chenyiteng -eo pid,ppid,pgid,stat,etime,args | grep -F 'fastwam-grpo-pi0style-offload-smoke1-2gpu32x8-g8-b2048-u2-m10-phys67-v1' | grep -v grep || true
echo '===NAMESPACES===' 
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os,ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_post_oom_probe', logging_level='ERROR')
from collections import Counter
rows=ray.util.list_named_actors(all_namespaces=True)
print(Counter(r.get('namespace') for r in rows))
for r in sorted(rows,key=lambda x:(x.get('namespace',''),x.get('name',''))):
    print(r.get('namespace'),r.get('name'))
ray.shutdown()
PY
echo '===GPU===' 
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo '===SHARED_RAY===' 
ps -p 321933 -o user,pid,stat,etime,args --no-headers
