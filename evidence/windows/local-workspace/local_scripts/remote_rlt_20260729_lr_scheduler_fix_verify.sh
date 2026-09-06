set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
CONFIG=examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml

test "$(git -C "$RLT_ROOT" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$RLT_ROOT" rev-parse HEAD)" = e4127fd49e38362161eac08c551a7a98c11e9802
sha256sum "$RLT_ROOT/$CONFIG"
git -C "$RLT_ROOT" diff --check -- "$CONFIG"
git -C "$RLT_ROOT" diff -- "$CONFIG"
