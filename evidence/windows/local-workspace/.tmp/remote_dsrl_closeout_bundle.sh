set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
OUT=/root/autodl-tmp/experiment_exports/dsrl_pi0_robotwin_formal_v1_20260729/dsrl_closeout_51753eab.bundle
BASE=acc7c14b93aec8eb2f2e8f32e4072be3957b761b
HEAD=51753eab66e49454f7ba5c56020cf49be706aba8

cd "$REPO"
test "$(git rev-parse HEAD)" = "$HEAD"
test "$(git rev-parse '@{upstream}')" = "$BASE"
test -z "$(git status --porcelain)"
git bundle create "$OUT" HEAD "^$BASE"
git bundle verify "$OUT"
echo "BUNDLE=$OUT"
stat -c 'SIZE=%s' "$OUT"
sha256sum "$OUT"
