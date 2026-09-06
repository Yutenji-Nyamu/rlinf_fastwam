set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
BRANCH=codex/dsrl-pi0-robotwin
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1

cd "$REPO"
echo "HOST=$(hostname)"
echo "TIME=$(date --iso-8601=seconds)"
echo "BRANCH=$(git branch --show-current)"
echo "HEAD=$(git rev-parse HEAD)"
echo "TRACKING=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
echo "LOCAL_DIVERGENCE=$(git rev-list --left-right --count HEAD...'@{upstream}')"
remote_line=$(timeout 30s git ls-remote personal "refs/heads/$BRANCH")
echo "REMOTE_LINE=$remote_line"
echo "RECENT_LOG_BEGIN"
git log --oneline --decorate -5
echo "RECENT_LOG_END"
if kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null; then
  echo "DSRL_DRIVER_ALIVE=1"
else
  echo "DSRL_DRIVER_ALIVE=0"
fi
