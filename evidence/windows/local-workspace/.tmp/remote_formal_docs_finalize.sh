set -euo pipefail

cd /root/autodl-tmp/RLinf_fastwam_rlinf
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = 1def9a24e46491f7801ad20badce6afd2fe81467
kill -0 70062

test "$(git status --short | wc -l)" -eq 2
git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 2
git commit -m "docs(dsrl): finalize formal status ledger"
git push

echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
test -z "$(git status --porcelain)"
echo "WORKTREE_CLEAN=1"
kill -0 70062
echo "DRIVER_ALIVE=1"
