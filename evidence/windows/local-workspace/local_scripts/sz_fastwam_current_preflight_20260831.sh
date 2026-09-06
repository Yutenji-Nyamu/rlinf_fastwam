set -euo pipefail

date '+TIME=%F %T %Z'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
free -h
df -h / /home /data
ps -eo user,pid,ppid,pgid,rss,etimes,args --sort=-rss | grep -E 'train_embodied_agent.py|ray::|raylet|gcs_server' | head -n 35 || true
