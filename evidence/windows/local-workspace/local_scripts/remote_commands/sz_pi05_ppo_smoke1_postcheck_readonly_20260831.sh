#!/usr/bin/env bash
set -u
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
RUN="$ROOT/runs/pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2"
CHAIN="$ROOT/smoke-chain-20260831-v1"

printf 'chain\n'
pid=$(cat "$CHAIN/worker.pid" 2>/dev/null || true)
printf 'pid=%s alive=%s\n' "$pid" "$(test -n "$pid" && kill -0 "$pid" 2>/dev/null && echo yes || echo no)"
printf 'completed='; test -f "$CHAIN/completed.txt" && tr '\n' ',' < "$CHAIN/completed.txt"; printf '\n'
printf 'chain_log_size='; stat -c %s "$CHAIN/chain.log" 2>/dev/null || true
tail -n 40 "$CHAIN/chain.log" 2>/dev/null || true

printf '\nrun_files\n'
find "$RUN" -maxdepth 8 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort
printf '\nresource_extrema\n'
awk -F, 'NR>1 && NF>=7 {if(min==0 || $3<min)min=$3; if($4>g2)g2=$4; if($6>g3)g3=$6} END {printf "min_available_kib=%s gpu2_peak_mib=%s gpu3_peak_mib=%s\n",min,g2,g3}' "$RUN/runtime/resource.csv"
printf '\nsave_markers\n'
grep -nEi 'save|checkpoint|Global Step|Traceback|ERROR|OutOfMemory|WorkerCrashed' "$RUN/runtime/driver.log" | tail -n 80 || true
printf '\nnext_run_dirs\n'
find "$ROOT/runs" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
