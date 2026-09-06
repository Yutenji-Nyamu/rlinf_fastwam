set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
branch=codex/sz-current-dsrl-pi0-robotwin

cd "$worktree"
test "$(git branch --show-current)" = "$branch"
git branch --set-upstream-to="personal/$branch" "$branch"

local_head=$(git rev-parse HEAD)
remote_head=$(git rev-parse "personal/$branch")
read -r ahead behind <<EOF
$(git rev-list --left-right --count "HEAD...personal/$branch")
EOF

test "$local_head" = "$remote_head"
test "$ahead" = 0
test "$behind" = 0
test -z "$(git status --porcelain)"

printf 'LOCAL_HEAD=%s\n' "$local_head"
printf 'REMOTE_HEAD=%s\n' "$remote_head"
printf 'AHEAD=%s\n' "$ahead"
printf 'BEHIND=%s\n' "$behind"
printf 'UPSTREAM=%s\n' "$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"
printf 'WORKTREE=CLEAN\n'
