set -eu
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
PID=$(cat "$RUN/runtime/wrapper.pid")
printf 'wrapper_pid=%s alive=' "$PID"
if kill -0 "$PID" 2>/dev/null; then echo yes; else echo no; fi
printf 'exit_code='
if test -s "$RUN/runtime/exit_code.txt"; then cat "$RUN/runtime/exit_code.txt"; else echo pending; fi
echo '== gpu45 =='
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '== log tail =='
tail -n 100 "$RUN/runtime/driver.log" 2>/dev/null || true
echo '== fatal scan =='
grep -Ein 'traceback|out of memory|CUDA error|RayActorError|WorkerCrashedError|nonfinite|nan|fatal' "$RUN/runtime/driver.log" 2>/dev/null | tail -n 20 || true
echo '== output files =='
find "$RUN" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -n 30
