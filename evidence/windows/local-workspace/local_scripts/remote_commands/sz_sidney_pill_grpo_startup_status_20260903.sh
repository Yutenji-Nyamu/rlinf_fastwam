#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
PID=$(<"$RUN/runtime/wrapper.pid")
printf 'timestamp=%s\nwrapper_pid=%s\nalive=' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$PID"
if kill -0 "$PID" 2>/dev/null; then echo yes; else echo no; fi
printf 'exit_code='; cat "$RUN/runtime/exit_code.txt" 2>/dev/null || echo pending
printf 'latest_step='; grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$RUN/runtime/driver.log" | tail -n1 || true
printf 'rollout_progress='; grep -aoE 'Rollout epoch:[[:space:]]+[0-9]+/4' "$RUN/runtime/driver.log" | tail -n1 || true
printf 'fatal_count='; grep -aEic 'Traceback|CUDA out of memory|RayActorError|NCCL.*error|non.?finite|Segmentation fault|Fatal Python error' "$RUN/runtime/driver.log" || true
printf 'task_mentions='; grep -ao 'move_pillbottle_pad' "$RUN/runtime/driver.log" | wc -l
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/python" - <<'PY'
import ray
ray.init(address='172.17.0.1:6389', namespace='codex_sidney_pill_status', logging_level='ERROR')
for ns in ('RLinf','RLinf_1'):
    print(f'{ns}_actors={sum(1 for x in ray.util.list_named_actors(all_namespaces=True) if x.get("namespace")==ns)}')
ray.shutdown()
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
tail -n 60 "$RUN/runtime/driver.log"
