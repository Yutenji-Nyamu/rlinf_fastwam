set -eu
date '+%F %T %Z'
nvidia-smi -i 3,4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
for s in 1000 1001 1002 1003 1004; do
  d=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/move_stapler_pad-official-valid-seed${s}-20260903
  [ -f "$d/eval.log" ] || continue
  step=$(grep -aoE '[0-9]+/1200' "$d/eval.log" | tail -1 || true)
  pc=$(grep -oE "'pc_success': [0-9.]+" "$d/eval.log" | tail -1 || true)
  ev=$(grep -oE "'eval_s': [0-9.]+" "$d/eval.log" | tail -1 || true)
  res=$(tr '\n' ' ' < "$d/result.txt" 2>/dev/null || true)
  printf 'seed=%s step=%s %s %s result=%s\n' "$s" "${step:-none}" "${pc:-running}" "${ev:-}" "$res"
done
ps -u chenyiteng -o pid,ppid,etimes,stat,args | grep -E 'lerobot-eval.*move_stapler_pad|sz_sidney_move_official_valid' | grep -v grep || true
