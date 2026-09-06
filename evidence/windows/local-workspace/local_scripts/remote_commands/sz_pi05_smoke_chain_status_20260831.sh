#!/usr/bin/env bash
set -u
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
CHAIN="$ROOT/smoke-chain-20260831-v1"
pid=$(cat "$CHAIN/worker.pid")
printf 'chain_pid=%s alive=%s\n' "$pid" "$(kill -0 "$pid" 2>/dev/null && echo yes || echo no)"
printf 'completed='; tr '\n' ',' < "$CHAIN/completed.txt" 2>/dev/null || true; printf '\n'
printf 'chain_tail\n'; tail -n 20 "$CHAIN/chain.log" 2>/dev/null || true
for name in \
  pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2 \
  pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2 \
  pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2; do
  run="$ROOT/runs/$name"
  test -d "$run" || continue
  printf '\nrun=%s\n' "$name"
  test -s "$run/runtime/exit_code.txt" && printf 'exit=' && cat "$run/runtime/exit_code.txt"
  tail -n 6 "$run/runtime/resource.csv" 2>/dev/null || true
  grep -E 'Global Step|global step|success|actor/|critic/|Traceback|CUDA out|OutOfMemory|WorkerCrashed|ERROR' "$run/runtime/driver.log" 2>/dev/null | tail -n 30 || true
  printf 'driver_tail\n'; tail -n 12 "$run/runtime/driver.log" 2>/dev/null || true
  find "$run/checkpoints" -maxdepth 4 -type f -printf '%s %p\n' 2>/dev/null | tail -n 10 || true
done
printf '\ngpus\n'
nvidia-smi -i 2,3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/^MemAvailable:/ {print "host_mem_available_kib=" $2}' /proc/meminfo
