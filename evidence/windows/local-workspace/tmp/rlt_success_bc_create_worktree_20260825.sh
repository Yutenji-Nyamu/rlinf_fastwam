set -euo pipefail
src=/root/autodl-tmp/RLinf_rlt_teacher_dvac
dst=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
branch=codex/rlt-dvac-success-episode-bc

test -e "$src/.git"
test ! -e "$dst"
git -C "$src" worktree add -b "$branch" "$dst" 74c715515c94fd367aff274871bf9488e95ff6b3
git -C "$dst" status --short --branch
git -C "$dst" rev-parse HEAD
