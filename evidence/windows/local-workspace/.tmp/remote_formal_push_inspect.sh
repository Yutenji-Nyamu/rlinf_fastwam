set -euo pipefail

date '+TIME=%Y-%m-%d %H:%M:%S %Z'
cd /root/autodl-tmp/RLinf_fastwam_rlinf
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS"
git status --short
echo "PUSH_PROCESSES"
pgrep -af 'git push|git-remote-https|remote_formal_docs_finalize' || true
kill -0 70062
echo "DRIVER_ALIVE=1"
