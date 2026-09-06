#!/usr/bin/env bash
set -euo pipefail

BASE=7d07a4212ee6858cc333e1d4fab7a37256d1f839
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
BRANCH=codex/sz-7d07a421-grpo-pi0-robotwin
CFG=examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml
EXPECTED_CFG_SHA=675245c1bbb5ba396db22896c042490f07938bfc42713e5589b12c4b527bce16

printf 'MARKER=SZ_GRPO_CURRENT_COMMIT_PUSH_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

test "$(git -C "$WT" rev-parse HEAD)" = "$BASE"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
test "$(sha256sum "$WT/$CFG" | awk '{print $1}')" = "$EXPECTED_CFG_SHA"
mapfile -t status_lines < <(git -C "$WT" status --porcelain=v1)
test "${#status_lines[@]}" = 1
test "${status_lines[0]}" = "?? $CFG"

identity_ref=personal/main
author_name="$(git -C "$WT" show -s --format=%an "$identity_ref")"
author_email="$(git -C "$WT" show -s --format=%ae "$identity_ref")"
test -n "$author_name"
test -n "$author_email"
for ref in \
  personal/codex/idea2-dvac-pi0-robotwin \
  personal/codex/idea2-dvac-train-weighting \
  personal/codex/idea2-dvac-residual-downweight
do
  test "$(git -C "$WT" show -s --format=%an "$ref")" = "$author_name"
  test "$(git -C "$WT" show -s --format=%ae "$ref")" = "$author_email"
done
export GIT_AUTHOR_NAME="$author_name"
export GIT_AUTHOR_EMAIL="$author_email"
export GIT_COMMITTER_NAME="$author_name"
export GIT_COMMITTER_EMAIL="$author_email"
printf 'git_identity_source=%s author=%s (process-only)\n' "$identity_ref" "$author_name"
printf 'personal_url=%s\n' "$(git -C "$WT" remote get-url personal)"

remote_before="$(git -C "$WT" ls-remote --heads personal "refs/heads/$BRANCH")"
test -z "$remote_before"
printf 'remote_branch_before=absent\n'

git -C "$WT" diff --no-index --check -- \
  "$WT/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml" \
  "$WT/$CFG" || test "$?" = 1
git -C "$WT" add -- "$CFG"
test "$(git -C "$WT" diff --cached --name-only)" = "$CFG"
expected_name_status="$(printf 'A\t%s' "$CFG")"
test "$(git -C "$WT" diff --cached --name-status)" = "$expected_name_status"
git -C "$WT" diff --cached --check
git -C "$WT" diff --cached --stat

git -C "$WT" commit \
  -m "feat(embodiment): add current pi0 RoboTwin GRPO recipe" \
  -m "Config-only port from current official base 7d07a421; keep current workers and schema."
commit="$(git -C "$WT" rev-parse HEAD)"
test "$(git -C "$WT" rev-parse HEAD^)" = "$BASE"
test -z "$(git -C "$WT" status --short)"
test "$(git -C "$WT" diff-tree --no-commit-id --name-only -r "$commit")" = "$CFG"
printf 'commit=%s\n' "$commit"
git -C "$WT" show --stat --oneline --decorate --no-renames "$commit"

timeout 60s git -C "$WT" push --set-upstream personal "HEAD:refs/heads/$BRANCH"
remote_after="$(git -C "$WT" ls-remote --heads personal "refs/heads/$BRANCH" | awk '{print $1}')"
test "$remote_after" = "$commit"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$commit"
test -z "$(git -C "$WT" status --short)"
printf 'remote_branch_after=%s\n' "$remote_after"
printf 'MARKER=SZ_GRPO_CURRENT_COMMIT_PUSH_OK\n'
