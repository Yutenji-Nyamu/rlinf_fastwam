#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
archive=/tmp/rlt_dvac_pure_committed_source_cb88e9c5.tar
test ! -e "$archive"
test "$(git -C "$repo" rev-parse HEAD)" = cb88e9c5d817a248fef6d6ee02127b874ab851a8
test -z "$(git -C "$repo" status --short)"
tar -cf "$archive" -C "$repo" \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s0p5_mb256_warm20k_replay80k_fresh480.yaml
sha256sum "$archive"
du -h "$archive"
