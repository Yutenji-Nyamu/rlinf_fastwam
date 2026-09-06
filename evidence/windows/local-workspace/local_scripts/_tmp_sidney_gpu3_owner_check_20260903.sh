set -eu
date '+%F %T %Z'
nvidia-smi -i 3 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits || true
nvidia-smi -i 3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
