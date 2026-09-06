set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
git diff -- \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
