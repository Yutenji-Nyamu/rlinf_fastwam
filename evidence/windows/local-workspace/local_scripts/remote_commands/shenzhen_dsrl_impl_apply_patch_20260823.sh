set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
patch_file=$(mktemp /tmp/rlinf-dsrl-current-patch.XXXXXX)
trap 'rm -f "$patch_file"' EXIT

cat > "$patch_file"

test "$(git -C "$worktree" rev-parse HEAD)" = "7d07a4212ee6858cc333e1d4fab7a37256d1f839"
test -z "$(git -C "$worktree" status --porcelain)"
git -C "$worktree" apply --check "$patch_file"
git -C "$worktree" apply "$patch_file"
git -C "$worktree" diff --check
git -C "$worktree" status --short
git -C "$worktree" diff --stat
