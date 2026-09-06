#!/usr/bin/env bash
set -euo pipefail

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
free -h
df -h / /home /data
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'raylet_count='; { pgrep -u "$(id -u)" -x raylet || true; } | wc -l
printf 'gcs_count='; { pgrep -u "$(id -u)" -x gcs_server || true; } | wc -l
printf 'rlt_head='; git -C /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421 rev-parse HEAD
printf 'rlt_dirty='; git -C /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421 status --porcelain | wc -l
printf 'dsrl_head='; git -C /data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin rev-parse HEAD
printf 'dsrl_dirty='; git -C /data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin status --porcelain | wc -l
for attempt in 1 2; do
  printf 'proxy_github_attempt%s=' "$attempt"
  curl -L --proxy http://127.0.0.1:7890 --max-time 15 -sS -o /dev/null -w '%{http_code}\n' https://github.com/ || true
  printf 'proxy_hf_attempt%s=' "$attempt"
  curl -L --proxy http://127.0.0.1:7890 --max-time 15 -sS -o /dev/null -w '%{http_code}\n' https://huggingface.co/ || true
done
printf 'direct_github_http='; curl -L --noproxy '*' --max-time 15 -sS -o /dev/null -w '%{http_code}\n' https://github.com/ || true
printf 'direct_hf_http='; curl -L --noproxy '*' --max-time 15 -sS -o /dev/null -w '%{http_code}\n' https://huggingface.co/ || true
