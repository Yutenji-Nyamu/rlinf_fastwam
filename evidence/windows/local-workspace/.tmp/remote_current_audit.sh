set -eu
hostname
pwd
id -u
date '+%F %T %Z'

cd /root/autodl-tmp/RLinf_fastwam_rlinf
git rev-parse HEAD
git branch --show-current
git status --short
git diff --name-status
git diff --stat
git remote -v

pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_020324-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu
test -d "$RUN"
tail -n 180 "$RUN/run_embodiment.log"
cat "$RUN/resource_monitor/peak.txt"
find "$RUN" -maxdepth 3 -type d -name 'global_step_*' -print | sort

test -d /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40/.git && git -C /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 status --short --branch || true
command -v zip || true
