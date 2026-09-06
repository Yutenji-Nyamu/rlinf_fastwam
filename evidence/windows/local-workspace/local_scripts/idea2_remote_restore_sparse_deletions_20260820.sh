set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
expected_head=6d0db56bf26f972cd27fa29535f5eb939e80e5bf
expected_branch=codex/idea2-dvac-pi0-robotwin

test "$(git -C "$target" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$target" branch --show-current)" = "$expected_branch"

before_count=$(git -C "$target" diff --name-only --diff-filter=D | wc -l)
printf 'TRACKED_DELETIONS_BEFORE=%s\n' "$before_count"
test "$before_count" -eq 314

git -C "$target" diff --name-only --diff-filter=D -z | \
  git -C "$target" restore --source=HEAD \
    --pathspec-from-file=- --pathspec-file-nul

after_count=$(git -C "$target" diff --name-only --diff-filter=D | wc -l)
test "$after_count" -eq 0
printf 'TRACKED_DELETIONS_AFTER=%s\n' "$after_count"
git -C "$target" diff --check
git -C "$target" status --short
git -C "$target" diff --stat

git -C /root/autodl-tmp/RLinf status --short
git -C /root/autodl-tmp/RLinf worktree list --porcelain | \
  grep -E '^(worktree|HEAD|branch) '
