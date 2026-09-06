#!/usr/bin/env bash
set +e

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' '=== identity ==='
hostname
id
printf '%s\n' '=== home guidance ==='
for file in /home/readme_to_codex.md /home/readme_storage_to_codex.md /home/readme_network_to_codex.md; do
  printf -- '--- %s ---\n' "$file"
  sed -n '1,220p' "$file" 2>&1 || true
done
printf '%s\n' '=== storage ==='
for target in / /home /data; do
  findmnt -T "$target" -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
done
df -hT / /home /data
df -ih / /home /data
du -sh \
  /home/chenyiteng/venvs \
  /home/chenyiteng/.cache \
  /data/chenyiteng/projects \
  /data/chenyiteng/models \
  /data/chenyiteng/results 2>/dev/null || true
printf '%s\n' '=== gpu and processes ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
pgrep -a -x raylet || true
pgrep -a -x gcs_server || true
pgrep -af 'train_embodied_agent.py|eval_embodied_agent.py|FastWAM|fastwam' || true
printf '%s\n' '=== proxy and small network probes ==='
systemctl is-active mihomo.service || true
systemctl is-enabled mihomo.service || true
ss -lnt | grep -E '127\.0\.0\.1:(7890|9090)' || true
source /etc/profile.d/mihomo-proxy.sh
curl -fsSIL --max-time 20 https://github.com | sed -n '1p'
curl -fsSIL --max-time 20 https://huggingface.co/api/models/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle | sed -n '1p'
printf 'proxy_http=%s\n' "${http_proxy:-unset}"
