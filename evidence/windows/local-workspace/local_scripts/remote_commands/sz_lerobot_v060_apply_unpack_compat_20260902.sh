set -euo pipefail
export GIT_PAGER=cat
dst=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
patch=/data/chenyiteng/projects/lerobot-sidney/lerobot_v060_py310_unpack_compat.patch
git -C "$dst" apply --recount --unidiff-zero --check "$patch"
git -C "$dst" apply --recount --unidiff-zero "$patch"
git -C "$dst" diff --check
git -C "$dst" diff --stat
/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python -m compileall -q "$dst/src/lerobot"
echo COMPILEALL_OK
sha256sum "$patch"
