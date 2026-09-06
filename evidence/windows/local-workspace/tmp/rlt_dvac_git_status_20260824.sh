set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'HEAD=' && git rev-parse HEAD
printf 'STATUS\n'
git status --short
printf 'UPSTREAM=' && git rev-list --left-right --count '@{upstream}...HEAD'
printf 'PUSH_PROCESS\n'
ps -eo pid,etimes,cmd | grep -E '[g]it push|[s]sh.*github' || true
