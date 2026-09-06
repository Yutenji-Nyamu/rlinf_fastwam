set -eu
date '+%F %T %Z'
nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
ps -u chenyiteng -o pid,ppid,etimes,stat,args | grep -E 'lerobot-eval|sz_sidney_(adjust|move)_official_valid' | grep -v grep || true
for s in 1003 1004 1008 1009; do
  d=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/adjust_bottle-official-valid-seed${s}-20260903
  if [ -f "$d/eval.log" ]; then
    printf 'seed=%s bytes=%s result=' "$s" "$(stat -c %s "$d/eval.log")"
    if [ -f "$d/result.txt" ]; then tr '\n' ' ' < "$d/result.txt"; else grep -aoE '[0-9]+/1200|running_success_rate=[0-9.]+%' "$d/eval.log" | tail -2 | tr '\n' ' '; fi
    echo
    grep -E "success_rate|sum_reward|avg_sum_reward|aggregated|Evaluation|eval_s" "$d/eval.log" | tail -6 || true
  fi
done
for s in 1000 1001 1002 1003 1004; do
  d=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/move_stapler_pad-official-valid-seed${s}-20260903
  if [ -f "$d/eval.log" ]; then
    printf 'move_seed=%s bytes=%s result=' "$s" "$(stat -c %s "$d/eval.log")"
    if [ -f "$d/result.txt" ]; then tr '\n' ' ' < "$d/result.txt"; else grep -aoE '[0-9]+/1200|running_success_rate=[0-9.]+%' "$d/eval.log" | tail -2 | tr '\n' ' '; fi
    echo
    grep -E "'pc_success':|'successes':|eval_s" "$d/eval.log" | tail -3 || true
  fi
done
