set -euo pipefail
base=/root/autodl-tmp/RLinf
target=/root/autodl-tmp/RLinf_qam_pi0_robotwin
commit=6d0db56bf26f972cd27fa29535f5eb939e80e5bf
branch=codex/qam-pi0-robotwin

printf 'PRECHECK\n'
test ! -e "$target"
test "$(git -C "$base" rev-parse "$commit^{commit}")" = "$commit"
test -z "$(git -C "$base" show-ref --verify --hash "refs/heads/$branch" || true)"
git -C "$base" status --short

git -C "$base" worktree add -b "$branch" "$target" "$commit"

printf 'POSTCHECK\n'
git -C "$target" branch --show-current
git -C "$target" rev-parse HEAD
git -C "$target" status --short
git -C "$base" worktree list --porcelain
