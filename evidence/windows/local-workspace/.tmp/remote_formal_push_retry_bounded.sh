set -euo pipefail

cd /root/autodl-tmp/RLinf_fastwam_rlinf
test "$(git rev-parse HEAD)" = 95e6251841f6d7256ee2c13de053d4618e02e00e
timeout --signal=TERM --kill-after=5s 45s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
