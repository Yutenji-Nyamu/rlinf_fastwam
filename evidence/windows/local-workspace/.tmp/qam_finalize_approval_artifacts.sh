set -eu
source_cfg=/root/autodl-tmp/qam_source_resolved_20260731_v1.yaml
smoke_cfg=/root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml
diff_file=/root/autodl-tmp/qam_source_to_qonly_smoke_20260731_v1.diff
monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
bash -n "$monitor"
sha256sum "$source_cfg" "$smoke_cfg" "$monitor"
status=0
diff -u "$source_cfg" "$smoke_cfg" >"$diff_file" || status=$?
test "$status" -eq 1
sha256sum "$diff_file"
wc -l "$source_cfg" "$smoke_cfg" "$diff_file"
date '+%F %T %Z'
pgrep -af 'raylet|gcs_server|dashboard|monitor.py|log_monitor.py' || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
