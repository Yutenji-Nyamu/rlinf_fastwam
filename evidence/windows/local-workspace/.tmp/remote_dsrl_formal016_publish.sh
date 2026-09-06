set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=51753eab66e49454f7ba5c56020cf49be706aba8
LEDGER=docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
! kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null
test "$(git status --porcelain | sed 's/^...//')" = "$LEDGER"
grep -q 'FORMAL-016' "$LEDGER"
git diff --check

git add -- "$LEDGER"
git diff --cached --check
test "$(git diff --cached --name-only)" = "$LEDGER"
git commit -m "docs(dsrl): record closeout publication"

timeout --signal=TERM --kill-after=5s 180s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin

head_now=$(git rev-parse HEAD)
test "$(git rev-parse '@{upstream}')" = "$head_now"
test -z "$(git status --porcelain)"
! kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null
echo "HEAD=$head_now"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS=CLEAN"
echo "DRIVER_ALIVE=0"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
