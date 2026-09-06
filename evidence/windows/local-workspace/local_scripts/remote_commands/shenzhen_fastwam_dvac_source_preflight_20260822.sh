#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
BASE=7faa71108368fbb3b6885649f112af607427a2d4

echo "TIME_UTC=$(date -u +%FT%TZ)"
echo "USER=$(id -un) UID=$(id -u) HOST=$(hostname)"
test -e "$WT/.git" || { echo "MISSING_WORKTREE=$WT"; exit 20; }
cd "$WT"
echo "PWD=$PWD"
echo "HEAD=$(git rev-parse HEAD)"
echo "BRANCH=$(git branch --show-current)"
echo "MERGE_BASE=$(git merge-base HEAD "$BASE")"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
echo "REMOTE_BEGIN"
git remote -v
echo "REMOTE_END"
echo "WORKTREES_BEGIN"
git worktree list --porcelain
echo "WORKTREES_END"
echo "TARGET_FILES_BEGIN"
for f in \
  src/fastwam/models/wan22/fastwam.py \
  experiments/robotwin/fastwam_policy/deploy_policy.py \
  configs/sim_robotwin.yaml; do
  test -f "$f"
  stat -c '%n bytes=%s mtime=%y' "$f"
  sha256sum "$f"
done
echo "TARGET_FILES_END"
echo "SYMBOLS_BEGIN"
if command -v rg >/dev/null 2>&1; then
  rg -n "def infer_action|class WorldActionRobotWinPolicy|def _infer_action_chunk|def _fill_action_queue|def step\(|def reset\(|skip_get_obs_within_replan|replan_steps|infer_action\(" \
    src/fastwam/models/wan22/fastwam.py \
    experiments/robotwin/fastwam_policy/deploy_policy.py \
    configs/sim_robotwin.yaml
else
  grep -nE "def infer_action|class WorldActionRobotWinPolicy|def _infer_action_chunk|def _fill_action_queue|def step\(|def reset\(|skip_get_obs_within_replan|replan_steps|infer_action\(" \
    src/fastwam/models/wan22/fastwam.py \
    experiments/robotwin/fastwam_policy/deploy_policy.py \
    configs/sim_robotwin.yaml
fi
echo "SYMBOLS_END"
echo "TEST_LAYOUT_BEGIN"
find tests experiments -maxdepth 3 -type f -iname '*test*.py' 2>/dev/null | sort | head -80 || true
echo "TEST_LAYOUT_END"

test "$(git rev-parse HEAD)" = "$BASE"
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git branch --show-current)" = codex/sz-fastwam-dvac-observe
echo FASTWAM_DVAC_SOURCE_PREFLIGHT_OK
