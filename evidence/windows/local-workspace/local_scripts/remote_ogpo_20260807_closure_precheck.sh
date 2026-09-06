#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = codex/ogpo-pi0-robotwin

printf 'CLOSURE_PRECHECK host=%s head=%s branch=%s\n' \
  "$(hostname)" \
  "$(git -C "$repo" rev-parse HEAD)" \
  "$(git -C "$repo" branch --show-current)"

cd "$repo"
sha256sum \
  examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml \
  rlinf/algorithms/ogpo/core.py \
  rlinf/config.py \
  rlinf/runners/embodied_runner.py \
  rlinf/workers/actor/fsdp_ogpo_policy_worker.py \
  rlinf/workers/env/env_worker.py \
  tests/workers/test_ogpo_checkpoint_sidecar.py

printf 'CLOSURE_STATUS\n'
git status --short
printf 'CLOSURE_GPU_PROCESSES\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
