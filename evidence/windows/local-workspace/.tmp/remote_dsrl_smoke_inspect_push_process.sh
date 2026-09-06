set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
echo "HEAD=$(git -C "$REPO" rev-parse HEAD)"
echo "UPSTREAM=$(git -C "$REPO" rev-parse '@{upstream}')"
git -C "$REPO" status --short
ps -eo pid,ppid,etimes,cmd | grep -E '[g]it.*push|[g]it-remote-https|[c]url.*github' || true
