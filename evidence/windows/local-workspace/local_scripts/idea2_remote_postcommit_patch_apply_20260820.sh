set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
test "$(git -C "$target" rev-parse HEAD)" = \
  6a75555841053001d0366291267efa78051cd82b
test "$(git -C "$target" branch --show-current)" = \
  codex/idea2-dvac-pi0-robotwin
test -z "$(git -C "$target" status --short)"
git -C "$target" apply --whitespace=error-all -
git -C "$target" diff --check
git -C "$target" status --short
