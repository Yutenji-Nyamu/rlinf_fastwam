set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin

date -Is
test "$(git -C "$RLT_ROOT" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$RLT_ROOT" rev-parse HEAD)" = e4127fd49e38362161eac08c551a7a98c11e9802
test -z "$(git -C "$RLT_ROOT" status --short)"
git -C "$RLT_ROOT" rev-list --left-right --count HEAD...@{upstream}
test ! -e \
  "$RLT_ROOT/docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729"
printf '%s\n' SYNC_PREFLIGHT_OK
