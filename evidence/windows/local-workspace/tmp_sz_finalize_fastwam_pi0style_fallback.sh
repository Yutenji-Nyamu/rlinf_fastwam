set -euo pipefail
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-pi0style-offload-smoke1-2gpu16x16-g8-b2048-u2-m10-phys67-v1
R="$RUN/runtime"
echo "NOW=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
echo '===EXIT===' 
for f in started_at.txt finished_at.txt exit_code.txt wrapper.pid owned.pgid observer.pid; do test -s "$R/$f" && { printf '%s=' "$f"; cat "$R/$f"; }; done
echo '===METRICS===' 
"$VENV/bin/python" - "$R/driver.log" <<'PY'
import re,sys
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
s=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',s)
for line in s.splitlines():
    if any(k in line for k in ['Global Step:','reward=','success_once=','success_at_end=','actor/approx_kl=','actor/clip_fraction=','actor/grad_norm=']):
        print(line)
PY
echo '===FATAL===' 
grep -aEin 'CUDA out of memory|out of memory|OOM|Traceback|Fatal|RuntimeError|NCCL|RayActorError|SIG(SEGV|ABRT)|killed' "$R/driver.log" | tail -n 30 || true
echo '===RESOURCE===' 
awk -F, 'NR==1{next} {if($3>g6){g6=$3;t6=$1};if($5>g7){g7=$5;t7=$1};if(min==0||$2<min){min=$2;tm=$1};n++} END{printf "samples=%d gpu6_peak_mib=%d at=%s gpu7_peak_mib=%d at=%s mem_available_min_kib=%d at=%s\n",n,g6,t6,g7,t7,min,tm}' "$R/resource_2s.csv"
echo '===CHECKPOINT_TREE===' 
find "$RUN" -type f \( -name '*.distcp' -o -name '.metadata' -o -name 'manifest.json' -o -name 'checkpoint_complete*' -o -name '*state*.pt' -o -name '*state*.pth' \) -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -k3
echo '===GLOBAL_STEP_DIRS===' 
find "$RUN" -type d -name 'global_step_1' -print
echo '===RUN_SIZE===' 
du -sh "$RUN"
echo '===OWNED_PROCESSES===' 
ps -u chenyiteng -eo pid,ppid,pgid,stat,etime,args | grep -F 'fastwam-grpo-pi0style-offload-smoke1-2gpu16x16-g8-b2048-u2-m10-phys67-v1' | grep -v grep || true
echo '===NAMESPACES===' 
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os,ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_fastwam_fallback_final_probe', logging_level='ERROR')
from collections import Counter
print(Counter(r.get('namespace') for r in ray.util.list_named_actors(all_namespaces=True)))
ray.shutdown()
PY
echo '===GPU===' 
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
echo '===RAM===' 
free -h
cat /proc/pressure/memory
echo '===SHARED_RAY===' 
ps -p 321933 -o user,pid,stat,etime,args --no-headers
