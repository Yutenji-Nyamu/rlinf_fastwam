#!/usr/bin/env bash
set -u

ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
CHAIN="$ROOT/smoke-chain-grpo-continuation-20260831-v2"
pid=$(cat "$CHAIN/worker.pid" 2>/dev/null || true)
printf 'chain_pid=%s alive=%s\n' "$pid" "$(test -n "$pid" && kill -0 "$pid" 2>/dev/null && echo yes || echo no)"
printf 'completed='; tr '\n' ',' < "$CHAIN/completed.txt" 2>/dev/null || true; printf '\n'
printf 'chain_tail\n'; tail -n 30 "$CHAIN/chain.log" 2>/dev/null || true

for spec in \
  'pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2|pi05_ppo_smoke1|1' \
  'pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2|pi05_grpo_smoke1|1' \
  'pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2|pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2|2'; do
  IFS='|' read -r name experiment step <<< "$spec"
  run="$ROOT/runs/$name"
  printf '\nrun=%s\n' "$name"
  printf 'exit='; cat "$run/runtime/exit_code.txt" 2>/dev/null || true
  printf 'times='; paste -sd, "$run/runtime/started_at.txt" "$run/runtime/finished_at.txt" 2>/dev/null || true
  awk -F, 'NR>1 && NF>=7 {if(!n||$4>g2)g2=$4;if(!n||$6>g3)g3=$6;if(!n||$3<m)m=$3;n=1} END{if(n)printf "resource_peak_gpu2_mib=%s peak_gpu3_mib=%s min_mem_available_kib=%s\n",g2,g3,m}' "$run/runtime/resource.csv" 2>/dev/null || true
  grep -E 'Global Step:|success_once=|approx_kl=|clip_fraction=|grad_norm=|dvac_' "$run/runtime/driver.log" 2>/dev/null | tail -n 80 || true
  actor="$run/$experiment/checkpoints/global_step_$step/actor"
  printf 'checkpoint_files\n'
  find "$actor" -maxdepth 2 -type f -printf '%s %f\n' 2>/dev/null | sort
done

printf '\ngpus\n'
nvidia-smi -i 2,3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/^MemAvailable:/ {print "host_mem_available_kib=" $2}' /proc/meminfo
printf 'pi05_processes\n'
ps -eo pid,ppid,pgid,etimes,args --sort=pid | grep -F '/pi05-robotwin-rl' | grep -v grep || true
