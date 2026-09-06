#!/usr/bin/env bash
set -euo pipefail

control=/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-control-phys2-1cycle-20260830-v2
pure=/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-pure04-phys3-1cycle-20260830-v2
action=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
st=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1

date --iso-8601=seconds
grep '^MemAvailable:' /proc/meminfo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

for run in "$control" "$pure"; do
  printf '\nRUN %s\n' "$run"
  if [[ ! -d "$run/runtime" ]]; then
    printf 'ABSENT\n'
    continue
  fi
  pid=$(<"$run/runtime/wrapper.pid")
  if kill -0 "$pid" 2>/dev/null; then alive=1; else alive=0; fi
  printf 'wrapper=%s alive=%s exit=%s\n' "$pid" "$alive" "$(cat "$run/runtime/exit_code.txt" 2>/dev/null || printf pending)"
  printf 'checkpoint_files='; find "$run" -path '*global_step_1*' -type f 2>/dev/null | wc -l
  grep -E 'global_step|update_step|critic_updates|actor_updates|episode_success|loss|Traceback|ERROR|Error|OOM|out of memory' "$run/runtime/driver.log" 2>/dev/null | tail -n 16 || true
  tail -n 12 "$run/runtime/driver.log" 2>/dev/null || true
done

printf '\nGRPO\n'
for run in "$action" "$st"; do
  pid=$(<"$run/runtime/wrapper.pid")
  if kill -0 "$pid" 2>/dev/null; then alive=1; else alive=0; fi
  printf '%s wrapper=%s alive=%s log_bytes=%s mtime=%s fatal=%s\n' \
    "$run" "$pid" "$alive" \
    "$(stat -c %s "$run/runtime/driver.log")" \
    "$(stat -c %Y "$run/runtime/driver.log")" \
    "$(grep -Eci 'Traceback|OOM|out of memory|worker.*died|RayActorError' "$run/runtime/driver.log" || true)"
done

printf '\nRAY_NAMESPACES\n'
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray list actors --filter 'state=ALIVE' --format=json 2>/dev/null \
  | /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -c 'import json,sys,collections; d=json.load(sys.stdin); print(dict(collections.Counter(x.get("ray_namespace") for x in d)))' || true
