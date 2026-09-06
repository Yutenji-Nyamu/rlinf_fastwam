set -euo pipefail
target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
cd "$target"
test "$(git rev-parse HEAD)" = "73da63f01d52290c3537f12fb4bfb55cfd82f94c"
test "$(git branch --show-current)" = "codex/idea2-dvac-pi0-robotwin"
test -z "$(git status --porcelain)"
git apply -
git status --short

