#!/usr/bin/env bash
set -euo pipefail
D=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/scene-fence-smoke-20260904
date -Is
tail -n 22 "$D/smoke.log"
test ! -e "$D/exit_code" || cat "$D/exit_code"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -u chenyiteng -o pid,ppid,etime,stat,%cpu,rss,args | grep -E 'smoke.py|train_embodied_agent.py' | cut -c1-300 || true
