set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=50ebc6780435b677fa507286e6252559fd6b9c79
EXPECTED_UPSTREAM=b01661e8a6b3ca1b883fb61d4ade9a467ffd84b5

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_UPSTREAM"
test -z "$(git status --porcelain)"
kill -0 "$(cat "$RUN/formal.pid")"

echo "PREFLIGHT=PASS"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"

