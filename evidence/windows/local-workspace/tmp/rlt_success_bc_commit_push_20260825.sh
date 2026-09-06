set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
branch=codex/rlt-dvac-success-episode-bc
cd "$repo"

git diff --check
git add \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_fresh480.yaml
git diff --cached --check
git diff --cached --stat
git commit -m 'feat(rlt): weight successful episode BC with teacher DVAC'
git push -u personal "$branch"
git rev-parse HEAD
git status --short --branch
