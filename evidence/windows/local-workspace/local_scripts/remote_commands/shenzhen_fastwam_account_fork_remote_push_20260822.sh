#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
ACCOUNT_KEY=/home/chenyiteng/.ssh/github_yutenji_account_ed25519
KNOWN=/home/chenyiteng/.ssh/github_rlinf_fastwam_known_hosts
EXPECTED_HEAD=c63dc9b5384d6637a93cc862dbe2815d0332801d
BRANCH=codex/sz-fastwam-dvac-observe
PERSONAL_URL=git@github.com:Yutenji-Nyamu/FastWAM.git
SSH_CMD="ssh -i $ACCOUNT_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o UserKnownHostsFile=$KNOWN -o StrictHostKeyChecking=yes"

printf 'MARKER=SZ_FASTWAM_ACCOUNT_FORK_REMOTE_PUSH_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id
test "$(id -un)" = chenyiteng
test -f "$ACCOUNT_KEY"
test -f "$KNOWN"
source /etc/profile.d/mihomo-proxy.sh
gh auth status --hostname github.com

printf '%s\n' '=== fork state/create ==='
if gh api repos/Yutenji-Nyamu/FastWAM --jq .full_name >/tmp/sz_fastwam_fork_name 2>/tmp/sz_fastwam_fork_probe.err; then
  printf 'fork_state=already_exists full_name=%s\n' "$(cat /tmp/sz_fastwam_fork_name)"
else
  printf 'fork_state=creating\n'
  gh api -X POST repos/yuantianyuan01/FastWAM/forks --jq .full_name
fi

fork_ready=0
for attempt in $(seq 1 20); do
  if GIT_SSH_COMMAND="$SSH_CMD" git ls-remote "$PERSONAL_URL" HEAD refs/heads/main >/tmp/sz_fastwam_fork_lsremote.out 2>/tmp/sz_fastwam_fork_lsremote.err; then
    fork_ready=1
    printf 'fork_ready_attempt=%s\n' "$attempt"
    cat /tmp/sz_fastwam_fork_lsremote.out
    break
  fi
  sleep 3
done
if [[ "$fork_ready" -ne 1 ]]; then
  cat /tmp/sz_fastwam_fork_lsremote.err >&2
  exit 41
fi

printf '%s\n' '=== exact local worktree guard ==='
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$WT" diff --name-only)"
test -z "$(git -C "$WT" diff --cached --name-only)"
expected_untracked="$(printf '%s\n' \
  third_party/RoboTwin/assets \
  third_party/RoboTwin/policy/fastwam_policy \
  third_party/RoboTwin/task_config | sort)"
actual_untracked="$(git -C "$WT" ls-files --others --exclude-standard | sort)"
test "$actual_untracked" = "$expected_untracked"
printf 'head=%s branch=%s\n' "$(git -C "$WT" rev-parse HEAD)" "$(git -C "$WT" branch --show-current)"
printf '%s\n' "$actual_untracked"

printf '%s\n' '=== repo-local account-key transport ==='
if git -C "$WT" remote get-url personal >/dev/null 2>&1; then
  test "$(git -C "$WT" remote get-url personal)" = "$PERSONAL_URL"
else
  git -C "$WT" remote add personal "$PERSONAL_URL"
fi
git -C "$WT" config core.sshCommand "$SSH_CMD"
git -C "$WT" fetch --prune personal

remote_head="$(GIT_SSH_COMMAND="$SSH_CMD" git ls-remote "$PERSONAL_URL" "refs/heads/$BRANCH" | awk '{print $1}')"
if [[ -n "$remote_head" && "$remote_head" != "$EXPECTED_HEAD" ]]; then
  printf 'refusing_nonmatching_remote_head=%s\n' "$remote_head" >&2
  exit 42
fi
if [[ "$remote_head" = "$EXPECTED_HEAD" ]]; then
  printf 'push_state=already_exact\n'
  git -C "$WT" branch --set-upstream-to="personal/$BRANCH" "$BRANCH"
else
  printf 'push_state=ordinary_create\n'
  git -C "$WT" push --set-upstream personal "$BRANCH:$BRANCH"
fi

printf '%s\n' '=== post-push verification ==='
test "$(GIT_SSH_COMMAND="$SSH_CMD" git ls-remote "$PERSONAL_URL" "refs/heads/$BRANCH" | awk '{print $1}')" = "$EXPECTED_HEAD"
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$WT" diff --name-only)"
test -z "$(git -C "$WT" diff --cached --name-only)"
test "$(git -C "$WT" ls-files --others --exclude-standard | sort)" = "$expected_untracked"
git -C "$WT" remote -v
printf 'repo_core_ssh_command=%s\n' "$(git -C "$WT" config --get core.sshCommand)"
git -C "$WT" status --short --branch
printf 'remote_branch_head=%s\n' "$EXPECTED_HEAD"
printf 'MARKER=SZ_FASTWAM_ACCOUNT_FORK_REMOTE_PUSH_OK\n'
