#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin

hostname
pwd
id -u
test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = \
  codex/ogpo-pi0-robotwin
git -C "$repo" status --short --branch
sha256sum \
  "$repo/rlinf/models/embodiment/openpi/openpi_ogpo.py" \
  "$repo/rlinf/models/embodiment/openpi/openpi_action_model.py" \
  "$repo/tests/embodiment/test_openpi_ogpo_adapter.py"
printf 'GPU_COMPUTE_PROCESSES\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
