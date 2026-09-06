set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
git push
git rev-list --left-right --count '@{upstream}...HEAD'
