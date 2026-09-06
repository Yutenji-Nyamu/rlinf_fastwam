set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
cd "$worktree"

git config --local user.name "Yutenji-Nyamu"
git config --local user.email "1842710211@qq.com"

test "$(git rev-parse HEAD)" = "7d07a4212ee6858cc333e1d4fab7a37256d1f839"
test "$(git branch --show-current)" = "codex/sz-current-dsrl-pi0-robotwin"
git diff --check

git add -- \
  examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi.yaml \
  rlinf/data/storage/replay/__init__.py \
  rlinf/data/storage/replay/dsrl_transition.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/actor/fsdp_sac_policy_worker.py \
  tests/unit_tests/test_dsrl_target_shadow_resume.py \
  tests/unit_tests/test_dsrl_transition_replay.py

test "$(git diff --cached --name-only | wc -l)" -eq 7
git diff --cached --check
git diff --cached --stat
git commit -m "feat(dsrl): port RoboTwin DSRL to current RLinf"
git push personal HEAD:refs/heads/codex/sz-current-dsrl-pi0-robotwin

git rev-parse HEAD
git status --short
