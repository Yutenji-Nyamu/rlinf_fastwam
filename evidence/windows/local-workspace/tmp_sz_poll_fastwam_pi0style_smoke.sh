set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-pi0style-offload-smoke1-2gpu32x8-g8-b2048-u2-m10-phys67-v1
RUNTIME="$RUN/runtime"
echo "NOW=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
if test -s "$RUNTIME/wrapper.pid"; then
  PID=$(cat "$RUNTIME/wrapper.pid")
  if test -d "/proc/$PID"; then
    echo "WRAPPER_ALIVE=1 PID=$PID"
    ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd -p "$PID" --no-headers || true
  else
    echo "WRAPPER_ALIVE=0 PID=$PID"
  fi
fi
for f in started_at.txt finished_at.txt exit_code.txt; do
  if test -s "$RUNTIME/$f"; then printf '%s=' "$f"; cat "$RUNTIME/$f"; fi
done
echo 'DRIVER_TAIL_BEGIN'
tail -n 100 "$RUNTIME/driver.log" 2>/dev/null || true
echo 'DRIVER_TAIL_END'
echo 'FATAL_SCAN_BEGIN'
grep -Ein 'CUDA out of memory|out of memory|OOM|Traceback|Fatal|RuntimeError|NCCL|RayActorError|SIG(SEGV|ABRT)|killed' "$RUNTIME/driver.log" 2>/dev/null | tail -n 40 || true
echo 'FATAL_SCAN_END'
echo 'RESOURCE_TAIL_BEGIN'
tail -n 8 "$RUNTIME/resource_2s.csv" 2>/dev/null || true
echo 'RESOURCE_TAIL_END'
echo 'RESOURCE_PEAK_BEGIN'
awk -F, 'NR==1{next} {if($3>g6)g6=$3;if($5>g7)g7=$5;if(min==0||$2<min)min=$2;n++} END{printf "samples=%d gpu6_peak_mib=%d gpu7_peak_mib=%d mem_available_min_kib=%d\n",n,g6,g7,min}' "$RUNTIME/resource_2s.csv" 2>/dev/null || true
echo 'RESOURCE_PEAK_END'
echo 'GPU67_BEGIN'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi -i 6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
echo 'GPU67_END'
echo 'OTHER_GPU_SUMMARY_BEGIN'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo 'OTHER_GPU_SUMMARY_END'
echo 'SHARED_RAY_BEGIN'
ps -p 321933 -o pid,stat,etime,cmd --no-headers || true
echo 'SHARED_RAY_END'
echo 'CHECKPOINTS_BEGIN'
find "$RUN" -maxdepth 4 -type f \( -name 'manifest.json' -o -name '*.distcp' -o -name '.metadata' -o -name 'checkpoint_complete*' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 30
echo 'CHECKPOINTS_END'
