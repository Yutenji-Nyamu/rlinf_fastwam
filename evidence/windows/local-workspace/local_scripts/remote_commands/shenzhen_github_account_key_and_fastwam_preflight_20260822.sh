#!/usr/bin/env bash
set -euo pipefail

ACCOUNT_KEY=/home/chenyiteng/.ssh/github_yutenji_account_ed25519
KNOWN=/home/chenyiteng/.ssh/github_rlinf_fastwam_known_hosts
FASTWAM_WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
ACCOUNT_EXPECTED='SHA256:gMSUm/WoenHcy4U1nBbunKPYpv4mCHgsgHfKR1TmK/w'
FASTWAM_EXPECTED=c63dc9b5384d6637a93cc862dbe2815d0332801d
SSH_CMD="ssh -i $ACCOUNT_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o UserKnownHostsFile=$KNOWN -o StrictHostKeyChecking=yes"

printf 'MARKER=SZ_GITHUB_ACCOUNT_KEY_FASTWAM_PREFLIGHT_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

test "$(id -un)" = chenyiteng
test -f "$ACCOUNT_KEY"
test -f "$ACCOUNT_KEY.pub"
test -f "$KNOWN"
actual_fingerprint="$(ssh-keygen -lf "$ACCOUNT_KEY.pub" -E sha256 | awk '{print $2}')"
printf 'account_key_fingerprint=%s\n' "$actual_fingerprint"
test "$actual_fingerprint" = "$ACCOUNT_EXPECTED"
stat -c 'key mode=%a owner=%U:%G path=%n' "$ACCOUNT_KEY" "$ACCOUNT_KEY.pub"
printf 'known_hosts_fingerprint='
ssh-keygen -lf "$KNOWN" -E sha256

printf '%s\n' '=== explicit account-key ssh authentication ==='
set +e
ssh_output="$($SSH_CMD -T git@github.com 2>&1)"
ssh_rc=$?
set -e
printf 'ssh_t_rc=%s\n%s\n' "$ssh_rc" "$ssh_output"
case "$ssh_output" in
  *"Hi Yutenji-Nyamu! You've successfully authenticated"*) ;;
  *) printf 'unexpected_github_identity\n' >&2; exit 31 ;;
esac

printf '%s\n' '=== account-owned repositories via explicit key ==='
GIT_SSH_COMMAND="$SSH_CMD" git ls-remote git@github.com:Yutenji-Nyamu/rlinf_fastwam.git HEAD refs/heads/main

set +e
GIT_SSH_COMMAND="$SSH_CMD" git ls-remote git@github.com:Yutenji-Nyamu/FastWAM.git HEAD refs/heads/main > /tmp/sz_fastwam_account_lsremote.out 2> /tmp/sz_fastwam_account_lsremote.err
fastwam_lsremote_rc=$?
set -e
printf 'fastwam_lsremote_rc=%s\n' "$fastwam_lsremote_rc"
sed -n '1,20p' /tmp/sz_fastwam_account_lsremote.out
sed -n '1,20p' /tmp/sz_fastwam_account_lsremote.err

printf '%s\n' '=== gh state ==='
if command -v gh >/dev/null 2>&1; then
  printf 'gh_path=%s\n' "$(command -v gh)"
  gh --version | sed -n '1,2p'
  set +e
  gh auth status --hostname github.com
  printf 'gh_auth_status_rc=%s\n' "$?"
  set -e
else
  printf 'gh_path=absent\n'
fi

printf '%s\n' '=== Fast-WAM worktree state ==='
test -d "$FASTWAM_WT/.git" -o -f "$FASTWAM_WT/.git"
test "$(git -C "$FASTWAM_WT" rev-parse HEAD)" = "$FASTWAM_EXPECTED"
printf 'head=%s\n' "$(git -C "$FASTWAM_WT" rev-parse HEAD)"
printf 'branch=%s\n' "$(git -C "$FASTWAM_WT" branch --show-current)"
git -C "$FASTWAM_WT" status --short --branch
git -C "$FASTWAM_WT" remote -v
printf 'repo_core_ssh_command=%s\n' "$(git -C "$FASTWAM_WT" config --get core.sshCommand || true)"

if [[ "$fastwam_lsremote_rc" -eq 0 ]]; then
  printf 'FASTWAM_FORK=EXISTS\n'
else
  printf 'FASTWAM_FORK=ABSENT_OR_INACCESSIBLE\n'
fi
printf 'MARKER=SZ_GITHUB_ACCOUNT_KEY_FASTWAM_PREFLIGHT_OK\n'
