#!/usr/bin/env bash
set -euo pipefail

pi05=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
fast=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1

for label in PI05 FASTWAM; do
  if [ "$label" = PI05 ]; then run=$pi05; else run=$fast; fi
  echo "=== $label ==="
  echo "RUN=$run"
  find "$run/runtime" -maxdepth 1 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %f\n' | sort
  for marker in exit_code timeout_exit_code complete.json; do
    if [ -f "$run/runtime/$marker" ]; then echo "$marker=$(cat "$run/runtime/$marker")"; fi
  done
  echo 'OWNED_PROCESSES'
  ps -eo pid,ppid,pgid,etimes,rss,cmd | grep -F "$run" | grep -v grep || true
  driver="$run/runtime/driver.log"
  echo 'GLOBAL_STEPS'
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$driver" | tail -20 || true
  echo 'SUCCESS_AND_OPT'
  grep -aE 'train/success|success_rate|actor/approx_kl|actor/clip_fraction|actor/grad_norm|eval.*success|fixed' "$driver" | tail -120 || true
  echo 'FIRST_FATAL_LINES'
  grep -anE 'Fatal Python error|Aborted|Segmentation fault|terminate called|what\(\)|ErrorInitializationFailed|vk::|CUDA out of memory|OutOfMemory|RayActorError|WorkerCrashedError|unavailable|Socket closed|Traceback|Exiting main process' "$driver" | head -100 || true
  echo 'DRIVER_TAIL120'
  tail -120 "$driver"
done

echo '=== FASTWAM_RAY_ACTOR_LOGS ==='
session=/data/chenyiteng/ray/rlt-dsrl-v3/session_latest/logs
for pid in 3590591 3590594; do
  echo "PID=$pid"
  find -L "$session" -maxdepth 1 -type f \( -name "*${pid}*.err" -o -name "*${pid}*.out" \) -print | while read -r f; do
    echo "FILE=$f"
    grep -anE 'Fatal Python error|Aborted|Segmentation fault|terminate called|what\(\)|ErrorInitializationFailed|vk::|CUDA|Traceback|Exception' "$f" | head -80 || true
    tail -80 "$f" || true
  done
done

echo '=== RAY_NAMESPACE_COUNTS ==='
for ns in RLinf RLinf_1; do
  count=$(/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - "$ns" <<'PY'
import ray, sys
ray.init(address="172.17.0.1:6389", namespace=sys.argv[1], ignore_reinit_error=True, logging_level="ERROR")
print(sum(1 for a in ray.util.list_named_actors(all_namespaces=True) if a.get("namespace") == sys.argv[1]))
ray.shutdown()
PY
)
  echo "$ns=$count"
done
