set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
git add \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
git commit -m 'fix(rlt): keep DVAC metric schema rank-stable'
git push
printf 'HEAD=' && git rev-parse HEAD
printf 'STATUS\n'
git status --short
printf 'UPSTREAM\n'
git rev-list --left-right --count '@{upstream}...HEAD'
