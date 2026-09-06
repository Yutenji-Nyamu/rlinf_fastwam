#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
NEW_CONTROL=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b512-eval5-phys45-v1
NEW_DVAC=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to2-formal100-2gpu64x4-b512-eval5-phys67-v1

TZ=Asia/Shanghai date --iso-8601=seconds
printf 'identity='; id
test "$(git -C "$WT" rev-parse HEAD)" = 0e28ac6f09f821ea12e7d54eba7118ce0000ca86
printf 'source_head='; git -C "$WT" rev-parse HEAD
printf 'source_status_begin\n'; git -C "$WT" status --short; printf 'source_status_end\n'
test ! -e "$NEW_CONTROL"
test ! -e "$NEW_DVAC"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
printf 'shared_ray=alive address=%s\n' "$RAY_ADDRESS"

pid=$(<"$RUN/runtime/wrapper.pid")
pgid=$(<"$RUN/runtime/owned.pgid")
test "$pid" = "$pgid"
kill -0 "$pid"
printf 'old_wrapper_pid=%s old_pgid=%s owner=%s\n' "$pid" "$pgid" "$(ps -o user= -p "$pid" | xargs)"
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/runtime/driver.log" | tail -n 8 || true
printf 'old_fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$RUN/runtime/driver.log" || true

echo '=== GPU 4-7 process ownership and Ray job ids ==='
for process_pid in $(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu); do
  [[ -r "/proc/$process_pid/environ" ]] || continue
  job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  printf 'pid=%s job=%s user=%s cmd=%s\n' "$process_pid" "${job:-none}" "$(ps -o user= -p "$process_pid" | xargs)" "$(ps -o comm= -p "$process_pid" | xargs)"
done

RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_dual_grpo_preflight", logging_level="ERROR")
for row in sorted(ray.util.list_named_actors(all_namespaces=True), key=lambda x: (x.get("namespace", ""), x.get("name", ""))):
    print(f"named_actor namespace={row.get('namespace')} name={row.get('name')}")
ray.shutdown()
PY

nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,3p'
df -h /data /home | sed -n '1,3p'
