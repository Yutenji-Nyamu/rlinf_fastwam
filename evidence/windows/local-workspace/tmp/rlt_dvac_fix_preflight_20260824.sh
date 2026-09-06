set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'HEAD=' && git rev-parse HEAD
printf 'BRANCH=' && git branch --show-current
printf 'STATUS\n'
git status --short
printf 'SOURCE_HASHES\n'
sha256sum \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
