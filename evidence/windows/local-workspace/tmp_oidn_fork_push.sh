set -euo pipefail
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
BRANCH=codex/sz-robotwin-vector-render-lifecycle-fix
HEAD=8c7380c118ce7ca8a4ea4df53d753adc8fab0df2

test "$(git -C "$RT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$RT" status --porcelain)"
if ! gh api repos/Yutenji-Nyamu/RoboTwin --silent >/dev/null 2>&1; then
  gh repo fork RoboTwin-Platform/RoboTwin --clone=false --remote=false
fi
for _ in $(seq 1 20); do
  gh api repos/Yutenji-Nyamu/RoboTwin --silent >/dev/null 2>&1 && break
  sleep 1
done
gh api repos/Yutenji-Nyamu/RoboTwin --jq '.full_name + " fork=" + (.fork|tostring)'
if git -C "$RT" remote get-url personal >/dev/null 2>&1; then
  test "$(git -C "$RT" remote get-url personal)" = https://github.com/Yutenji-Nyamu/RoboTwin.git
else
  git -C "$RT" remote add personal https://github.com/Yutenji-Nyamu/RoboTwin.git
fi
git -C "$RT" push -u personal "$BRANCH:$BRANCH"
test "$(git -C "$RT" rev-parse HEAD)" = "$(git -C "$RT" rev-parse "personal/$BRANCH")"
git -C "$RT" status --short --branch
