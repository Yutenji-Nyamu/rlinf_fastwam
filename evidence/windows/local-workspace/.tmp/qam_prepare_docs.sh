set -euo pipefail
target=/root/autodl-tmp/RLinf_qam_pi0_robotwin
test "$(git -C "$target" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$target" branch --show-current)" = codex/qam-pi0-robotwin
mkdir -p "$target/docs/rlinf-robotwin-pi0-qam/evidence"
