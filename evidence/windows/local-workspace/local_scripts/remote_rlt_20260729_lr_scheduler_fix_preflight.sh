set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
CONFIG=examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml

test "$(git -C "$RLT_ROOT" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$RLT_ROOT" rev-parse HEAD)" = e4127fd49e38362161eac08c551a7a98c11e9802
git -C "$RLT_ROOT" diff --quiet -- "$CONFIG"
test "$(sha256sum "$RLT_ROOT/$CONFIG" | awk '{print $1}')" = \
  0fa01fa8c6f8624438a3d27288ecb848336cd2857599bc4b1a1d369dfc563cb3
printf '%s\n' LR_SCHEDULER_FIX_PREFLIGHT_OK
