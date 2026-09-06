set -euo pipefail

cd /root/autodl-tmp/RLinf_fastwam_rlinf
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = 1def9a24e46491f7801ad20badce6afd2fe81467
kill -0 70062
git push
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
