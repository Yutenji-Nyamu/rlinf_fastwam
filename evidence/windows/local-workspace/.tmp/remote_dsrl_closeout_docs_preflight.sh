set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
CKPT=$RUN/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1/checkpoints/global_step_195
ARCHIVE=/root/autodl-tmp/experiment_exports/dsrl_pi0_robotwin_formal_v1_20260729/dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz
EXPECTED_HEAD=acc7c14b93aec8eb2f2e8f32e4072be3957b761b
EXPECTED_ARCHIVE_SHA=f64762c1f95f881732facf9d7da2870dc1bfb96f9a2e4328180d145b8a7f877c

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
driver=$(cat "$RUN/formal.pid")
! kill -0 "$driver" 2>/dev/null
test "$(find "$CKPT" -type f | wc -l)" -eq 11
test -z "$(find "$CKPT" \( -name '*.tmp' -o -name '.metadata.tmp' \))"
test "$(sha256sum "$ARCHIVE" | awk '{print $1}')" = "$EXPECTED_ARCHIVE_SHA"

echo "PREFLIGHT=PASS"
echo "HEAD=$(git rev-parse HEAD)"
echo "DRIVER_ALIVE=0"
echo "DCP195_BYTES=$(du -sb "$CKPT" | awk '{print $1}')"
echo "ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")"

