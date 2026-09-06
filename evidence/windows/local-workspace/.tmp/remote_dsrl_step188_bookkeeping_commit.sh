set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=dc6a3a430be9c3a5002b436c4aeeaa399509f334

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
kill -0 "$(cat "$RUN/formal.pid")"

echo "PREFLIGHT=PASS"
echo "LAST_STEP_BEFORE=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"
