set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
cd "$repo"
sha256sum \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_teacher_dvac_w05to15_fresh480.yaml
