#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo/runs
CONTROL=$ROOT/ppo-control-smoke1-2gpu64x4-b1024-noeval-localshard-phys23-v1
DVAC=$ROOT/ppo-dvac-action-adv-fix-w0to2-smoke2-2gpu64x4-b1024-noeval-localshard-phys23-v1

TZ=Asia/Shanghai date --iso-8601=seconds
for run in "$CONTROL" "$DVAC"; do
  printf 'RUN=%s\n' "$run"
  if test ! -d "$run"; then printf 'ABSENT\n'; continue; fi
  runtime=$run/runtime
  pid=$(cat "$runtime/wrapper.pid" 2>/dev/null || true)
  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then printf 'WRAPPER_ALIVE=%s\n' "$pid"; else printf 'WRAPPER_DEAD=%s\n' "$pid"; fi
  if test -f "$runtime/exit_code.txt"; then printf 'EXIT='; cat "$runtime/exit_code.txt"; fi
  if test -f "$runtime/driver.log"; then
    stat -c 'LOG_BYTES=%s LOG_MTIME=%y' "$runtime/driver.log"
    grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:|Traceback|Error executing job|CUDA out of memory|OutOfMemory|RayActorError|WorkerCrashedError|Saving checkpoint|Saved checkpoint|dvac_weight|grad_norm' "$runtime/driver.log" | tail -n 80 || true
  fi
  find "$run" -maxdepth 6 -type d -name 'global_step_*' -printf 'CKPT=%p\n' 2>/dev/null | sort -V || true
done
printf 'GPU\n'
nvidia-smi -i 2,3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'NAMESPACES\n'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_ppo_smoke_status", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
for ns in sorted({str(x.get("namespace", "")) for x in rows if str(x.get("namespace", "")).startswith("RLinf")}):
    names = sorted(x["name"] for x in rows if x.get("namespace") == ns)
    print(ns, len(names), names)
ray.shutdown()
PY
