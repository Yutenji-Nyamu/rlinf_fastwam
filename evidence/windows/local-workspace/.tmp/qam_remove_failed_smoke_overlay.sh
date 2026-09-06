set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
target="$repo/examples/embodiment/config/robotwin_adjust_bottle_qam_openpi_qonly_smoke.yaml"
test -d "$repo/.git" || test -f "$repo/.git"
test -f "$target"
rm -- "$target"
test ! -e "$target"
printf 'REMOVED_OWN_UNTRACKED_OVERLAY=%s\n' "$target"
