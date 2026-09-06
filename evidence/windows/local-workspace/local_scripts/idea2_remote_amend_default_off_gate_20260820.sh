set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
old_commit=6a75555841053001d0366291267efa78051cd82b

cd "$target"
test "$(git rev-parse HEAD)" = "$old_commit"
test "$(git branch --show-current)" = codex/idea2-dvac-pi0-robotwin
test "$(git status --short)" = " M rlinf/workers/env/env_worker.py"
test "$(git diff --cached --name-only | wc -l)" -eq 0
git diff --check
git add -- rlinf/workers/env/env_worker.py
test "$(git diff --cached --name-only)" = \
  rlinf/workers/env/env_worker.py
git diff --cached --check
git diff --cached --stat
git commit --amend --no-edit
printf 'OLD_COMMIT=%s\n' "$old_commit"
printf 'AMENDED_COMMIT=%s\n' "$(git rev-parse HEAD)"
test -z "$(git status --short)"
printf 'POST_AMEND_CLEAN=1\n'
