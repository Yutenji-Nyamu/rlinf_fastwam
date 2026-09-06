set -u
date --iso-8601=seconds
for run in \
  '/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1' \
  '/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2'
do
  echo "=== RUN $run ==="
  echo '--- metrics tail ---'
  tail -n 80 "$run/metrics.log" 2>/dev/null || true
  echo '--- resource tail ---'
  tail -n 5 "$run/runtime/resource.csv" 2>/dev/null || true
  echo '--- runtime markers ---'
  find "$run/runtime" -maxdepth 1 -type f \( -name '*exit*' -o -name '*finished*' -o -name '*failed*' \) -printf '%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null || true
  echo '--- latest checkpoint/eval directories ---'
  find "$run" -maxdepth 5 -type d \( -name 'global_step_*' -o -path '*/video/eval/*' \) -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 12
done

echo '=== GPU_NOW ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits

