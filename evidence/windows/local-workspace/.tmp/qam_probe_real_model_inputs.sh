set -euo pipefail
date -Is
hostname
ps -eo pid,ppid,etime,%cpu,%mem,rss,args --sort=-rss |
  grep -E 'train_embodied_agent|ray::|raylet|gcs_server|qam' |
  grep -v grep || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
checkpoint=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
test "$(git -C "$repo" branch --show-current)" = codex/qam-pi0-robotwin
test -d "$checkpoint"
du -sh "$checkpoint"
find "$repo/examples/embodiment/config" -maxdepth 1 -type f \
  -name '*adjust_bottle*openpi*.yaml' -printf '%f\n' |
  sort
find /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/embodiment/config \
  -maxdepth 1 -type f -name '*adjust_bottle*openpi*.yaml' \
  -printf '%f\n' |
  sort
find /root/autodl-tmp/RLinf_fastwam_rlinf/examples/embodiment/config \
  -maxdepth 1 -type f -name '*adjust_bottle*openpi*.yaml' \
  -printf '%f\n' |
  sort
