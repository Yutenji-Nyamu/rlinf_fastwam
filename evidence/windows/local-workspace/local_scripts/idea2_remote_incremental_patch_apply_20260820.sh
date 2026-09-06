set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
test "$(git -C "$target" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$target" branch --show-current)" = \
  codex/idea2-dvac-pi0-robotwin
test "$(git -C "$target" diff --name-only --diff-filter=D | wc -l)" -eq 0
git -C "$target" apply --whitespace=error-all -
git -C "$target" diff --check
git -C "$target" status --short
