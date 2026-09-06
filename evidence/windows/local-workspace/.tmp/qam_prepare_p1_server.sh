set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
source_dir=/root/autodl-tmp/oracles/qam-2726d767
venv_dir=/root/autodl-tmp/venvs/qam-oracle-2726d767

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = codex/qam-pi0-robotwin
test ! -e "$source_dir"
test ! -e "$venv_dir"

mkdir -p \
  "$repo/rlinf/algorithms/qam" \
  "$repo/tests/algorithms/qam/oracle" \
  /root/autodl-tmp/oracles \
  /root/autodl-tmp/venvs
