set -euo pipefail
for pgid in 420388 420389 423903; do
  kill -KILL -- "-$pgid" 2>/dev/null || true
done
sleep 2
ps -eo pid,pgid,stat,cmd | grep -E '420388|420389|423752|423753|423903|train_embodied_agent|raylet|gcs_server' | grep -v grep | head -40 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
