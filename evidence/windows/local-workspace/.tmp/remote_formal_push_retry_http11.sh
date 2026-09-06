set -euo pipefail

cd /root/autodl-tmp/RLinf_fastwam_rlinf
test "$(git rev-parse HEAD)" = 95e6251841f6d7256ee2c13de053d4618e02e00e
test "$(git rev-parse '@{upstream}')" = 1def9a24e46491f7801ad20badce6afd2fe81467
git -c http.version=HTTP/1.1 push
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
test -z "$(git status --porcelain)"
echo "WORKTREE_CLEAN=1"
kill -0 70062
echo "DRIVER_ALIVE=1"
