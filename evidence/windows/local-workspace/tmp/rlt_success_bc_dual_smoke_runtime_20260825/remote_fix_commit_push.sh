set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
cd "$repo"
git diff --check
git add rlinf/algorithms/rlt/transition.py rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py tests/unit_tests/test_rlt_dvac_weighting.py
git commit -m "fix(rlt): keep replay next observations schema-stable"
git push personal HEAD:codex/rlt-dvac-success-episode-bc
git status --short
git rev-parse HEAD
