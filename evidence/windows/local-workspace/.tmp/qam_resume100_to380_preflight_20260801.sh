set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
ckpt=/root/autodl-tmp/experiments/qam_formal_resume25_to100_20260801_v3/robotwin_adjust_bottle_qam_formal_resume25_to100_20260801_v3/checkpoints/global_step_100

hostname
pwd
id -u
date '+time=%F %T %Z'
printf 'head='; git -C "$repo" rev-parse HEAD
test -z "$(git -C "$repo" status --short)"
printf 'training_or_ray\n'
pgrep -af '[t]rain_embodied_agent.py|[r]aylet|[g]cs_server' || true
printf 'checkpoint\n'
tr -d '\n' < "$ckpt/actor/qam_components/complete.json"
printf '\n'
test -f "$ckpt/actor/dcp_checkpoint/.metadata"
test -z "$(find "$ckpt" -name '.tmp-*' -o -name '*.tmp')"
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'disk_available_bytes='
df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d '[:space:]'
printf '\n'
printf 'oom\n'
awk '/oom |oom_kill / {print}' /sys/fs/cgroup/memory.events
