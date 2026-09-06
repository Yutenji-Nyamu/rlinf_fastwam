set -euo pipefail

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
OUT="$RUN_ROOT/fixed_latent_parity_v2"
pid=$(cat "$OUT/driver.pid")
monitor_pid=$(cat "$OUT/resource_monitor/monitor.pid")

if kill -0 "$pid" 2>/dev/null; then
  echo "PARITY_RUNNING=1"
else
  echo "PARITY_RUNNING=0"
fi
if kill -0 "$monitor_pid" 2>/dev/null; then
  echo "PARITY_MONITOR_RUNNING=1"
else
  echo "PARITY_MONITOR_RUNNING=0"
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
if test -s "$OUT/resource_monitor/peak.txt"; then
  cat "$OUT/resource_monitor/peak.txt"
fi
tail -n 80 "$OUT/driver.log"
