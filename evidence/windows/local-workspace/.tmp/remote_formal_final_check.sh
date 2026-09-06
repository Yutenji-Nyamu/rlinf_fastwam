set -euo pipefail

date '+TIME=%Y-%m-%d %H:%M:%S %Z'
kill -0 70062
echo "DRIVER_ALIVE=1"
cd /root/autodl-tmp/RLinf_fastwam_rlinf
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
test -z "$(git status --porcelain)"
echo "WORKTREE_CLEAN=1"
