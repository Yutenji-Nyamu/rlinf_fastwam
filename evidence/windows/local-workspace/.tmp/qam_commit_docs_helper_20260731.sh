set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git branch --show-current)" = codex/qam-pi0-robotwin

status="$(git status --short)"
printf '%s\n' "$status"
unexpected="$(
  printf '%s\n' "$status" |
    awk 'NF {print substr($0,4)}' |
    grep -Ev \
      '^(HANDOFF\.md|PROJECT_CONTEXT\.md|docs/rlinf-robotwin-pi0-qam/|local_scripts/)$' \
      || true
)"
test -z "$unexpected"

git add -- \
  PROJECT_CONTEXT.md \
  HANDOFF.md \
  local_scripts/remote_exec_autodl.py \
  docs/rlinf-robotwin-pi0-qam
git diff --cached --check
git diff --cached --stat
git commit -m "chore(qam): publish implementation docs and ssh helper"
echo "COMMIT=$(git rev-parse HEAD)"
test -z "$(git status --short)"

