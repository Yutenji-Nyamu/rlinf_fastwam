set -euo pipefail
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
BRANCH=codex/sz-robotwin-vector-render-lifecycle-fix
HEAD=8c7380c118ce7ca8a4ea4df53d753adc8fab0df2
test "$(git -C "$RT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$RT" status --porcelain)"
test "$(git -C "$RT" remote get-url personal)" = https://github.com/Yutenji-Nyamu/RoboTwin.git
gh auth setup-git
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
timeout 60s git -C "$RT" -c http.version=HTTP/1.1 push -u personal "$BRANCH:$BRANCH"
test "$(git -C "$RT" rev-parse HEAD)" = "$(git -C "$RT" rev-parse "personal/$BRANCH")"
git -C "$RT" status --short --branch
