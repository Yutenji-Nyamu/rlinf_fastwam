set -u
PI05=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
FAST=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2
date '+TIME %Y-%m-%d %H:%M:%S %Z'
printf 'FAST_PROGRESS\n'
grep -anE 'Generating Rollout Epochs|Global Step:|Fatal|Traceback|OOM|out of memory' "$FAST/runtime/driver.log" 2>/dev/null | tail -20 || true
printf 'PI05_PROGRESS\n'
grep -anE 'Global Step:|Fatal|Traceback|OOM|out of memory' "$PI05/runtime/driver.log" 2>/dev/null | tail -12 || true
printf 'GPU_4_7\n'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'MEMORY\n'
free -h
printf 'FAST_EXIT\n'
test -f "$FAST/runtime/exit_code" && cat "$FAST/runtime/exit_code" || printf 'none\n'
