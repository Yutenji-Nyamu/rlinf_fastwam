set -euo pipefail
printf '%s\n' '--- process ---'
pgrep -af '/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/lerobot-eval|lerobot.scripts.lerobot_eval' || true
printf '%s\n' '--- gpu4 ---'
nvidia-smi -i 4 --query-gpu=memory.used,utilization.gpu --format=csv,noheader
printf '%s\n' '--- gpu4 compute processes ---'
nvidia-smi -i 4 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
