set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
BASE=554c6dc8d586162d9444c01fa88308ed4f5203d0
BRANCH=codex/sz-fastwam-current-rlinf-grpo

test "$(git -C "$WT" rev-parse HEAD)" = "$BASE"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
git -C "$WT" diff --check
git -C "$WT" add -A
git -C "$WT" diff --cached --check
git -C "$WT" commit -m "feat(embodiment): add current Fast-WAM RoboTwin GRPO"
HEAD=$(git -C "$WT" rev-parse HEAD)
git -C "$WT" push -u personal "$BRANCH"
test -z "$(git -C "$WT" status --porcelain)"
REMOTE=$(git -C "$WT" ls-remote personal "refs/heads/$BRANCH" | awk '{print $1}')
test "$HEAD" = "$REMOTE"
printf 'HEAD=%s\nREMOTE=%s\nBRANCH=%s\nFASTWAM_COMMIT_PUSH_OK\n' "$HEAD" "$REMOTE" "$BRANCH"
