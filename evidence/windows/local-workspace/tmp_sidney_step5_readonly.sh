set -euo pipefail
run=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
echo PROCESSES
ps -eo pid,etimes,stat,args | grep -F "$run" | grep -v grep || true
echo EXIT
find "$run" -maxdepth 3 -type f \( -iname '*exit*' -o -iname '*status*' \) -print -exec tail -n 5 {} \; 2>/dev/null || true
echo METRICS_AND_LOG_TAIL
find "$run" -maxdepth 4 -type f \( -name '*.log' -o -name '*.out' -o -name '*.jsonl' -o -name '*.csv' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -20 | cut -d' ' -f2- | while IFS= read -r f; do
  echo "--- $f"
  grep -aEi 'fixed|eval|success|step.?5|fatal|traceback|error|oom|out of memory' "$f" 2>/dev/null | tail -n 40 || true
done
