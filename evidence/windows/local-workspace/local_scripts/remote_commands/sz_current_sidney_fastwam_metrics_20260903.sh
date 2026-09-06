set -eu

TZ=Asia/Shanghai date '+TIME=%Y-%m-%d %H:%M:%S %Z'

inspect_metrics() {
  label="$1"
  run="$2"
  echo "=== ${label} ==="
  echo "run=${run}"
  driver="$run/runtime/driver.log"
  [ -f "$driver" ] || { echo 'driver_missing'; return 0; }

  echo 'last_global_steps:'
  grep -a 'Global Step:' "$driver" | tail -n 4 || true

  echo 'last_train_successes:'
  grep -aoE 'success_once=[0-9]+([.][0-9]+)?' "$driver" | tail -n 10 || true
  grep -aoE 'success_once=[0-9]+([.][0-9]+)?' "$driver" | tail -n 10 | \
    awk -F= '{v[++n]=$2} END {if(n){s5=0;c5=(n<5?n:5);for(i=n-c5+1;i<=n;i++)s5+=v[i];s10=0;for(i=1;i<=n;i++)s10+=v[i];printf("ma5=%.6f ma10_available_%d=%.6f\n",s5/c5,n,s10/n)}}' || true

  echo 'eval_lines:'
  grep -aiE 'fixed.?32|eval.*success|success.*eval|evaluation.*(success|step)|Evaluating Rollout Epochs: 100%' "$driver" | tail -n 12 || true

  echo 'fatal_counts:'
  for pat in 'CUDA out of memory' 'OIDN Error' 'pthread_key_create' 'PyGILState_Release' 'ErrorInitializationFailed' 'ActorDiedError' 'WorkerCrashedError' 'Traceback'; do
    count=$(grep -aic "$pat" "$driver" 2>/dev/null || true)
    echo "$pat=$count"
  done

  echo 'checkpoint_dirs:'
  find "$run" -type d -name 'global_step_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 6 | cut -d' ' -f2- || true
  echo 'checkpoint_files:'
  find "$run" -type f \( -name '.metadata' -o -name '*.distcp' -o -name 'manifest.json' \) -printf '%s %p\n' 2>/dev/null | tail -n 12 || true
}

SIDNEY='/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'
FAST='/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2'
inspect_metrics SIDNEY "$SIDNEY"
inspect_metrics FASTWAM "$FAST"

echo '=== GPU4_7 ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | sed -n '5,8p'
