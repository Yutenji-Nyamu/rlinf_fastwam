#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-w0to2-smoke2-2gpu64x4-b1024-noeval-phys23-v1
RUNTIME="$RUN/runtime"
pid=$(cat "$RUNTIME/wrapper.pid" 2>/dev/null || true)
printf 'time='; TZ=Asia/Shanghai date --iso-8601=seconds
printf 'wrapper_pid=%s alive=' "$pid"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo yes; else echo no; fi
printf 'exit_code='; cat "$RUNTIME/exit_code.txt" 2>/dev/null || echo pending
awk '/^MemAvailable:/ {print "host_mem_available_kib=" $2}' /proc/meminfo
nvidia-smi -i 2,3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' '--- progress ---'
grep -E 'global step|Global Step|Step [0-9]+|Train.*success|dvac_|Saving checkpoint|checkpoint|Traceback|ERROR|Error|RuntimeError|OutOfMemory' "$RUNTIME/driver.log" 2>/dev/null | tail -n 50 || true
printf '%s\n' '--- tail ---'
tail -n 30 "$RUNTIME/driver.log" 2>/dev/null || true
printf '%s\n' '--- dvac artifacts/checkpoint ---'
find "$RUN" -maxdepth 6 -type f \( -name 'runner_step_*.pt' -o -name 'manifest.json' -o -name 'dvac_state_rank*.json' -o -name 'exit_code.txt' \) -printf '%p %s\n' 2>/dev/null | sort || true
