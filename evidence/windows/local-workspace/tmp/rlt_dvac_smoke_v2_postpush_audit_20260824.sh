set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'HEAD='
git rev-parse HEAD
printf 'UPSTREAM='
git rev-parse '@{upstream}'
printf 'AHEAD_BEHIND='
git rev-list --left-right --count '@{upstream}...HEAD'
git status --short
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'CGROUP_CURRENT='
cat /sys/fs/cgroup/memory.current
printf 'OOM_EVENTS\n'
grep -E '^(oom|oom_kill) ' /sys/fs/cgroup/memory.events
printf 'TRAIN_PROCESSES\n'
pgrep -af 'rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2|ray::|raylet|gcs_server' || true
