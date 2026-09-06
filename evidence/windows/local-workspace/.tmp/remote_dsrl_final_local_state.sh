set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
DCP=$RUN/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1/checkpoints/global_step_195
ARCHIVE=/root/autodl-tmp/experiment_exports/dsrl_pi0_robotwin_formal_v1_20260729/dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz

cd "$REPO"
echo "BRANCH=$(git branch --show-current)"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
if test -z "$(git status --porcelain)"; then
  echo "STATUS=CLEAN"
else
  echo "STATUS=DIRTY"
  git status --short
fi
if kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null; then
  echo "DRIVER_ALIVE=1"
else
  echo "DRIVER_ALIVE=0"
fi
echo "DCP195_BYTES=$(du -sb "$DCP" | awk '{print $1}')"
echo "DCP195_FILES=$(find "$DCP" -type f | wc -l)"
echo "RUNTIME_ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")"
echo "RUNTIME_ARCHIVE_SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
