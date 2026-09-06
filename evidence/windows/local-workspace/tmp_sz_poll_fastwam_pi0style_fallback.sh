set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-pi0style-offload-smoke1-2gpu16x16-g8-b2048-u2-m10-phys67-v1
R="$RUN/runtime"
echo "NOW=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
PID=$(cat "$R/wrapper.pid")
if test -d "/proc/$PID"; then echo "WRAPPER=alive PID=$PID"; else echo "WRAPPER=dead PID=$PID"; fi
for f in started_at.txt finished_at.txt exit_code.txt; do test -s "$R/$f" && { printf '%s=' "$f"; cat "$R/$f"; }; done
echo 'PROGRESS:'
grep -aE 'Generating Rollout Epochs:|Global Step:|Training Step|train/|success|Success|kl|clip|grad|Finished|saving|checkpoint' "$R/driver.log" 2>/dev/null | tail -n 60 || true
echo 'FATAL:'
grep -aEin 'CUDA out of memory|out of memory|OOM|Traceback|Fatal|RuntimeError|NCCL|RayActorError|SIG(SEGV|ABRT)|killed' "$R/driver.log" 2>/dev/null | tail -n 30 || true
echo 'RESOURCE:'
awk -F, 'NR==1{next} {if($3>g6){g6=$3;t6=$1};if($5>g7){g7=$5;t7=$1};if(min==0||$2<min){min=$2;tm=$1};n++} END{printf "samples=%d gpu6_peak_mib=%d at=%s gpu7_peak_mib=%d at=%s mem_available_min_kib=%d at=%s\n",n,g6,t6,g7,t7,min,tm}' "$R/resource_2s.csv" 2>/dev/null || true
tail -n 3 "$R/resource_2s.csv" 2>/dev/null || true
echo 'GPU67:'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo 'GPU67_PROCESSES:'
nvidia-smi -i 6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
echo 'OTHER_GPU:'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo 'FILES:'
find "$RUN" -maxdepth 5 -type f \( -name 'manifest.json' -o -name '*.distcp' -o -name '.metadata' -o -name 'checkpoint_complete*' \) -printf '%s %p\n' 2>/dev/null | sort | tail -n 15
