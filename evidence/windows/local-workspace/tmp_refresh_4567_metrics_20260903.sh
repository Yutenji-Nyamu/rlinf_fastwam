set -u
SID=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
FAST=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2
for pair in "SIDNEY:$SID" "FASTWAM:$FAST"; do
  name=${pair%%:*}
  dir=${pair#*:}
  echo "===== $name ====="
  echo 'METRICS_TAIL'
  tail -n 180 "$dir/metrics.log" 2>/dev/null || true
  echo 'DRIVER_PROGRESS'
  grep -E 'Global Step|Generating Rollout Epochs: 100%|Evaluation|eval.*success|success_rate|fixed' "$dir/runtime/driver.log" 2>/dev/null | tail -n 40 || true
  echo 'FATAL_COUNT'
  grep -Eic 'OIDN Error|pthread_key_create failed|PyGILState|CUDA out of memory|Traceback|WorkerCrashedError|NCCL.*(error|timeout)|exit_code=255' "$dir/runtime/driver.log" 2>/dev/null || true
  echo 'CHECKPOINTS'
  find "$dir" -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 5 || true
done
