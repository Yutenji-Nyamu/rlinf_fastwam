#!/usr/bin/env bash
set -u

ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
RUN="$ROOT/runs/pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2"
CHAIN="$ROOT/smoke-chain-20260831-v1"

printf 'chain_log_bytes='; stat -c %s "$CHAIN/chain.log" 2>/dev/null || true
printf 'chain_log\n'; sed -n '1,160p' "$CHAIN/chain.log" 2>/dev/null || true
printf 'run_files\n'
find "$RUN" -maxdepth 6 -type f -printf '%s %p\n' 2>/dev/null | sort
printf 'run_dirs\n'
find "$RUN" -maxdepth 5 -type d -printf '%p\n' 2>/dev/null | sort
printf 'checkpoint_log_lines\n'
grep -nEi 'checkpoint|saving|saved|save' "$RUN/runtime/driver.log" 2>/dev/null | tail -n 80 || true
printf 'pi05_processes\n'
ps -eo pid,ppid,pgid,etimes,args --sort=pid | grep -F '/pi05-robotwin-rl' | grep -v grep || true
