set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=4447d40211be8c78874bf4b000c871e7fbd93561
EXPECTED_UPSTREAM=b01661e8a6b3ca1b883fb61d4ade9a467ffd84b5

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_UPSTREAM"
kill -0 "$(cat "$RUN/formal.pid")"

expected_paths=$(cat <<'EOF'
HANDOFF.md
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
EOF
)
actual_paths=$(git status --porcelain | sed 's/^...//' | sort)
test "$actual_paths" = "$expected_paths"
git diff --check

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 2

git commit -m "docs(dsrl): record step 53 push blocker"

echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
kill -0 "$(cat "$RUN/formal.pid")"
echo "DRIVER_ALIVE=1"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"
