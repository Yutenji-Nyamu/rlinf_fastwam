set -euo pipefail
export GIT_PAGER=cat
base=/data/chenyiteng/projects/lerobot-sidney
repo=$base/lerobot-30da8e687a6d
dst=$base/lerobot-v060-py310-compat
patch=$base/lerobot_v060_py310_compat.patch
if [ ! -e "$dst" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" worktree add -q --detach "$dst" 30da8e687a6dfc617fcd94afc367ac7071c376ce
fi
test "$(git -C "$dst" rev-parse HEAD)" = 30da8e687a6dfc617fcd94afc367ac7071c376ce
if git -C "$dst" apply --recount --check "$patch"; then
  git -C "$dst" apply --recount "$patch"
fi
echo '=== diff ==='
git -C "$dst" diff --check
git -C "$dst" diff --stat
git -C "$dst" diff
echo '=== compileall ==='
/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python -m compileall -q "$dst/src/lerobot"
echo COMPILEALL_OK
sha256sum "$patch"
