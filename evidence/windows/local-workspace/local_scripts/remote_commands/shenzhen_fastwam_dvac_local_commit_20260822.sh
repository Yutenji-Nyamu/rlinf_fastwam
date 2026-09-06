#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
BASE=7faa71108368fbb3b6885649f112af607427a2d4
cd "$WT"

echo "TIME_UTC=$(date -u +%FT%TZ)"
test "$(git rev-parse HEAD)" = "$BASE"
test "$(git branch --show-current)" = codex/sz-fastwam-dvac-observe
git diff --cached --check

expected_paths="$(printf '%s\n' \
  configs/sim_robotwin.yaml \
  experiments/robotwin/eval_robotwin_single.py \
  experiments/robotwin/fastwam_policy/deploy_policy.py \
  experiments/robotwin/fastwam_policy/deploy_policy.yml \
  experiments/robotwin/fastwam_policy/dvac_telemetry.py \
  src/fastwam/models/wan22/fastwam.py \
  tests/test_fastwam_dvac_telemetry.py | sort)"
actual_paths="$(git diff --cached --name-only | sort)"
test "$actual_paths" = "$expected_paths"

echo "COMMIT_IDENTITY=Yutenji-Nyamu <1842710211@qq.com> (process-local only)"
staged_tree="$(git write-tree)"
echo "STAGED_TREE=$staged_tree"
git -c user.name=Yutenji-Nyamu -c user.email=1842710211@qq.com \
  commit -m "feat: add opt-in Fast-WAM DVAC telemetry"

commit="$(git rev-parse HEAD)"
parent="$(git rev-parse HEAD^)"
tree="$(git rev-parse HEAD^{tree})"
test "$parent" = "$BASE"
test "$tree" = "$staged_tree"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git diff-tree --no-commit-id --name-only -r HEAD | sort)" = "$expected_paths"

echo "COMMIT=$commit"
echo "PARENT=$parent"
echo "TREE=$tree"
git status --short --branch
git show --stat --oneline --decorate --no-renames HEAD
echo FASTWAM_DVAC_LOCAL_COMMIT_OK
