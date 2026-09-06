set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1

cd "$REPO"
echo "BRANCH=$(git branch --show-current)"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM_TRACKING=$(git rev-parse '@{upstream}')"
echo "REMOTE_REF=$(git ls-remote personal refs/heads/codex/dsrl-pi0-robotwin | awk '{print $1}')"
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
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
