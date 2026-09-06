set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
branch=codex/qam-pi0-robotwin
cd "$repo"
test "$(git rev-parse HEAD)" = 6aa4ec95d51cab3a5f890317386d941a83bd70db
test -z "$(git status --short)"
install -m 0644 /root/autodl-tmp/qam_docs_sync_20260801_v4/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
git diff --check
git add -- docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
git diff --cached --check
git commit -m "docs(qam): complete cycle 380 launch ledger"
local_head=$(git rev-parse HEAD)
(
  source /etc/network_turbo >/dev/null 2>&1
  GIT_TERMINAL_PROMPT=0 timeout 60 git push personal "HEAD:$branch"
  remote_head=$(timeout 20 git ls-remote personal "refs/heads/$branch" | awk '{print $1}')
  printf 'local_head=%s\nremote_head=%s\n' "$local_head" "$remote_head"
  test "$local_head" = "$remote_head"
)
test -z "$(git status --short)"
