set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin

cd "$target"
printf 'SERVER_TIME=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %:z')"
printf 'HEAD=%s\n' "$(git rev-parse HEAD)"
printf 'BRANCH=%s\n' "$(git branch --show-current)"
printf '%s\n' '--- REMOTE NAMES ---'
git remote
printf '%s\n' '--- STATUS ---'
git status --short
printf '%s\n' '--- CHANGED PATHS ---'
git status --short | awk '{print $2}' | sort
printf 'TRACKED_DELETIONS=%s\n' "$(git diff --name-only --diff-filter=D | wc -l)"
printf 'STAGED_PATHS=%s\n' "$(git diff --cached --name-only | wc -l)"
git diff --check

test "$(git rev-parse HEAD)" = 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git branch --show-current)" = codex/idea2-dvac-pi0-robotwin
test "$(git diff --name-only --diff-filter=D | wc -l)" -eq 0
test "$(git diff --cached --name-only | wc -l)" -eq 0
test "$(git status --short | wc -l)" -eq 6
printf 'PRECOMMIT_AUDIT_PASS=1\n'
