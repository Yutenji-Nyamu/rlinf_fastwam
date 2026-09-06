#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
expected_patch_sha=dae5cbb31d01a6a114f91b34c9e19ecf01b341d03eb01cb57d912a79ec3299d0
expected=(
  examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml
  examples/embodiment/train_embodied_agent.py
  rlinf/algorithms/ogpo/__init__.py
  rlinf/algorithms/ogpo/core.py
  rlinf/config.py
  rlinf/data/ogpo_replay.py
  rlinf/models/embodiment/base_policy.py
  rlinf/models/embodiment/modules/ogpo_critic.py
  rlinf/models/embodiment/modules/ogpo_modules.py
  rlinf/models/embodiment/openpi/__init__.py
  rlinf/models/embodiment/openpi/openpi_action_model.py
  rlinf/models/embodiment/openpi/openpi_ogpo.py
  rlinf/runners/embodied_runner.py
  rlinf/workers/actor/fsdp_ogpo_policy_worker.py
  rlinf/workers/env/env_worker.py
  rlinf/workers/rollout/hf/huggingface_worker.py
  tests/algorithms/test_ogpo_core.py
  tests/data/test_ogpo_replay.py
  tests/embodiment/ogpo_fsdp_ema_fixture.py
  tests/embodiment/ogpo_real_fsdp_ema_probe.py
  tests/embodiment/ogpo_real_fsdp_update_probe.py
  tests/embodiment/test_ogpo_critic.py
  tests/embodiment/test_openpi_ogpo_adapter.py
  tests/workers/test_ogpo_checkpoint_sidecar.py
  tests/workers/test_ogpo_env_trace.py
  tests/workers/test_ogpo_row_schedule.py
)

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = \
  codex/ogpo-pi0-robotwin
test -z "$(git -C "$repo" diff --cached --name-only)"

git -C "$repo" add -- "${expected[@]}"
git -C "$repo" diff --cached --check HEAD
test -z "$(git -C "$repo" diff --name-only)"
test -z "$(git -C "$repo" ls-files --others --exclude-standard)"
actual_patch_sha=$(
  git -C "$repo" diff --cached --binary --full-index HEAD \
    | sha256sum \
    | awk '{print $1}'
)
test "$actual_patch_sha" = "$expected_patch_sha"
printf 'STAGED_PATCH_SHA256=%s\n' "$actual_patch_sha"
git -C "$repo" diff --cached --stat HEAD
git -C "$repo" status --short --untracked-files=all
