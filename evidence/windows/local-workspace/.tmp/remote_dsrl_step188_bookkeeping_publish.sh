set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=dc6a3a430be9c3a5002b436c4aeeaa399509f334

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
kill -0 "$(cat "$RUN/formal.pid")"

expected_paths=$(cat <<'EOF'
HANDOFF.md
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
EOF
)
actual_paths=$(git status --porcelain | sed 's/^...//' | sort)
test "$actual_paths" = "$expected_paths"
git diff --check
grep -q 'dc6a3a430be9c3a5002b436c4aeeaa399509f334' HANDOFF.md
grep -q 'FORMAL-014.1' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 2
git commit -m "docs(dsrl): record step 188 publication"

timeout --signal=TERM --kill-after=5s 55s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin

head_now=$(git rev-parse HEAD)
test "$(git rev-parse '@{upstream}')" = "$head_now"
test -z "$(git status --porcelain)"
kill -0 "$(cat "$RUN/formal.pid")"

echo "HEAD=$head_now"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS=CLEAN"
echo "DRIVER_ALIVE=1"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"
