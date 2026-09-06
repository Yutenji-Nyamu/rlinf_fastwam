set -eu

echo '=== TIME ==='
TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z'

echo '=== GPU 4-7 ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | sed -n '5,8p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true

echo '=== ACTIVE CHENYITENG RLINF ==='
ps -u chenyiteng -o pid,ppid,pgid,etimes,stat,args --sort=pid | grep -E 'fastwam|pi05-sidney|main_embodied|wrapper\.sh|resource_observer' | grep -v grep | tail -n 120 || true

echo '=== RECENT SIDNEY RUNS ==='
find /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 5 | cut -d' ' -f2-

echo '=== RECENT FASTWAM RUNS ==='
find /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 8 | cut -d' ' -f2-

inspect_run() {
  run="$1"
  [ -d "$run" ] || return 0
  echo "--- RUN $run"
  if [ -f "$run/runtime/wrapper.pid" ]; then
    pid=$(cat "$run/runtime/wrapper.pid")
    if kill -0 "$pid" 2>/dev/null; then echo "wrapper_alive=$pid"; else echo "wrapper_dead=$pid"; fi
  fi
  [ -f "$run/runtime/exit_code.txt" ] && echo "exit_code=$(cat "$run/runtime/exit_code.txt")"
  echo 'latest_steps:'
  grep -a 'Global Step:' "$run/runtime/driver.log" 2>/dev/null | tail -n 4 || true
  echo 'recent_progress:'
  grep -aE 'Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint|save_checkpoint|fixed|Evaluation' "$run/runtime/driver.log" 2>/dev/null | tail -n 16 || true
  echo 'fatal_scan:'
  grep -aiE 'Traceback|CUDA out of memory|OutOfMemoryError|OIDN Error|pthread_key_create|PyGILState_Release|ErrorInitializationFailed|NCCL.*(error|failed)|ActorDiedError|SYSTEM_ERROR|WorkerCrashedError|Exiting main process due' "$run/runtime/driver.log" 2>/dev/null | tail -n 16 || true
  echo 'checkpoints:'
  find "$run/checkpoints" "$run/ckpt" "$run/checkpoint" -mindepth 1 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 8 | cut -d' ' -f2- || true
}

SIDNEY=$(find /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs -mindepth 1 -maxdepth 1 -type d -name 'move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1' -print -quit 2>/dev/null || true)
FAST=$(find /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs -mindepth 1 -maxdepth 1 -type d -iname '*renderer*fix*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
if [ -z "$FAST" ]; then
  FAST=$(find /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
fi

echo '=== SIDNEY DETAIL ==='
[ -n "$SIDNEY" ] && inspect_run "$SIDNEY"
echo '=== FASTWAM DETAIL ==='
[ -n "$FAST" ] && inspect_run "$FAST"
