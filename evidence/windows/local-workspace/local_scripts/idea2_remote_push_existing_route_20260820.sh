set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
branch=codex/idea2-dvac-pi0-robotwin
commit=6a75555841053001d0366291267efa78051cd82b

cd "$target"
test "$(git rev-parse HEAD)" = "$commit"
test "$(git branch --show-current)" = "$branch"
test -z "$(git status --short)"
printf '%s\n' '--- REMOTE BEFORE USING EXISTING ROUTE ---'
timeout 30s git ls-remote personal "refs/heads/$branch"
timeout 60s git push personal "HEAD:refs/heads/$branch"
printf '%s\n' '--- REMOTE AFTER ---'
remote_after="$(timeout 30s git ls-remote personal "refs/heads/$branch")"
printf '%s\n' "$remote_after"
test "$(printf '%s\n' "$remote_after" | awk '{print $1}')" = "$commit"
printf 'PUSH_VERIFIED=1\n'
