set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
ruff=/root/autodl-tmp/RLinf/.venv/bin/ruff
cd "$repo"
"$ruff" format \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
git diff --check
git diff --stat
