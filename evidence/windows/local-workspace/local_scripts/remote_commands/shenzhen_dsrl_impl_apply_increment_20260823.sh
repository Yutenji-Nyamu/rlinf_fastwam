set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
patch_file=$(mktemp /tmp/rlinf-dsrl-current-increment.XXXXXX)
trap 'rm -f "$patch_file"' EXIT
cat > "$patch_file"
git -C "$worktree" apply --check "$patch_file"
git -C "$worktree" apply "$patch_file"
git -C "$worktree" diff --check
