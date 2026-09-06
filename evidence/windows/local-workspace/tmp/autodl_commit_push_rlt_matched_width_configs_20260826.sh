#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
cd "$repo"

echo SEED_HASHES
sha256sum \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin/rlinf/envs/robotwin/seeds/train_seeds.json \
  "$repo/rlinf/envs/robotwin/seeds/train_seeds.json" \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin/rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json \
  "$repo/rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json"

control=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_mb256_warm20k_replay80k_control.yaml
method=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_mb256_warm20k_replay80k_gpu1_fresh480.yaml
git status --short
git add -- "$control" "$method"
git diff --cached --check
git commit -m "config: match single-GPU RLT aggregate width"
git push origin codex/rlt-dvac-success-episode-bc
echo NEW_HEAD
git rev-parse HEAD
git status --short
