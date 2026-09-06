#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
test "$(git -C "$root" branch --show-current)" = codex/sz-pi0-online-bc
test "$(git -C "$root" rev-parse HEAD)" = dc9b87cc49334c7516487ead68ebeb060fd7c090
test -z "$(git -C "$root" diff --cached --name-only)"
files=(rlinf/data/online_bc.py rlinf/workers/actor/fsdp_online_bc_policy_worker.py rlinf/models/embodiment/openpi/openpi_action_model.py examples/embodiment/train_embodied_agent.py rlinf/workers/rollout/hf/huggingface_worker.py rlinf/workers/env/env_worker.py examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml tests/unit_tests/test_online_bc.py docs/online_bc.md)
git -C "$root" diff --check
git -C "$root" add -- "${files[@]}"
git -C "$root" diff --cached --check
git -C "$root" diff --cached --stat
git -C "$root" commit -m 'Add chunk-aligned pi0 online success BC with single-GPU capacity smoke'
timeout 50s git -C "$root" push -u personal codex/sz-pi0-online-bc
git -C "$root" rev-parse HEAD
timeout 25s git -C "$root" ls-remote personal refs/heads/codex/sz-pi0-online-bc
git -C "$root" status --short
