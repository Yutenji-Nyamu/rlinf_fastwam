set -u

echo 'WRAPPERS'
ps -p 2660157,2812508 -o pid=,etime=,stat=,args= 2>/dev/null || true

echo 'SIDNEY_DIRS'
find /data/chenyiteng/results/rlinf-shenzhen -maxdepth 5 -type d -iname '*sidney*' 2>/dev/null | tail -n 20 || true

echo 'RECENT_DRIVER_FILES'
find /data/chenyiteng/results/rlinf-shenzhen -type f \( -iname '*driver*.log' -o -iname '*driver*.out' -o -name 'driver.stdout' -o -name 'run.log' \) -mmin -240 -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 30 || true

echo 'FASTWAM_TAIL'
FAST=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2
find "$FAST" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 20 || true
grep -R -E 'Global Step|success_rate|eval.*success|OIDN|pthread_key_create|PyGILState|CUDA out of memory|Traceback|WorkerCrashed|exit_code' "$FAST" --include='*.log' --include='*.out' --include='*.txt' 2>/dev/null | tail -n 120 || true
