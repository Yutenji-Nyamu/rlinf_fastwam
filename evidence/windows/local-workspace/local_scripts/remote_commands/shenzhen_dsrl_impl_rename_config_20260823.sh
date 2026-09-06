set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
old_path="$worktree/examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi_sz_h100_4gpu.yaml"
new_path="$worktree/examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi.yaml"
patch_file=$(mktemp /tmp/rlinf-dsrl-config-topology.XXXXXX)
trap 'rm -f "$patch_file"' EXIT
cat > "$patch_file"

if test -f "$old_path" && test ! -e "$new_path"; then
  mv "$old_path" "$new_path"
elif test -f "$new_path" && test ! -e "$old_path"; then
  :
else
  echo "Unexpected DSRL config rename state" >&2
  exit 1
fi
git -C "$worktree" apply --check "$patch_file"
git -C "$worktree" apply "$patch_file"
git -C "$worktree" diff --check
git -C "$worktree" status --short
