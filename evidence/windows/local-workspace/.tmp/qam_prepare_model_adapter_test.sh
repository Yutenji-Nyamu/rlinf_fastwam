set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = codex/qam-pi0-robotwin
mkdir -p \
  "$repo/rlinf/models/embodiment/modules" \
  "$repo/rlinf/models/embodiment/openpi" \
  "$repo/tests/embodiment"
