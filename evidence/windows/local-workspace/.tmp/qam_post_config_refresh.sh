set -eu
date '+%F %T %Z'
pgrep -af 'raylet|gcs_server|dashboard|monitor.py|log_monitor.py' || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
sha256sum /root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml
stat --format='%s %y %n' \
  /root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml
