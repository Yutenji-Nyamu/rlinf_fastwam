#!/usr/bin/env bash
set -euo pipefail

wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
expected=7d07a4212ee6858cc333e1d4fab7a37256d1f839

echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST=$(hostname) USER=$(id -un) UID=$(id -u)"
test -d "$wt/.git" -o -f "$wt/.git"
cd "$wt"
echo "CWD=$PWD"
echo "HEAD=$(git rev-parse HEAD)"
echo "BRANCH=$(git branch --show-current)"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
echo "REMOTES_BEGIN"
git remote -v
echo "REMOTES_END"
echo "WORKTREES_BEGIN"
git worktree list --porcelain
echo "WORKTREES_END"
test "$(git rev-parse HEAD)" = "$expected"
test -z "$(git status --porcelain)"
git cat-file -e 61996e15cc7f5a32bd6012b61b20893d94636c82^{commit}
echo "OLD_TELEMETRY_COMMIT_PRESENT=1"
for f in \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/env/env_worker.py; do
  test -f "$f"
  printf 'FILE %s LINES=' "$f"
  wc -l < "$f"
done
