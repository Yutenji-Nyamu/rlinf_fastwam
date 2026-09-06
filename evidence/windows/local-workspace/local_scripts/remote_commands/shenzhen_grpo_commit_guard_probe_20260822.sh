#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
BRANCH=codex/sz-7d07a421-grpo-pi0-robotwin
CFG=examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml

printf 'MARKER=SZ_GRPO_COMMIT_GUARD_PROBE_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
printf 'head=%s\n' "$(git -C "$WT" rev-parse HEAD)"
printf 'branch=%s\n' "$(git -C "$WT" branch --show-current)"
printf 'cfg_sha=%s\n' "$(sha256sum "$WT/$CFG" | awk '{print $1}')"
printf '%s\n' '=== STATUS QUOTED ==='
git -C "$WT" status --porcelain=v1 | sed -n l
if git -C "$WT" config --get user.name >/dev/null; then printf 'has_user_name=yes\n'; else printf 'has_user_name=no\n'; fi
if git -C "$WT" config --get user.email >/dev/null; then printf 'has_user_email=yes\n'; else printf 'has_user_email=no\n'; fi
printf 'personal_url=%s\n' "$(git -C "$WT" remote get-url personal)"
printf '%s\n' '=== REMOTE SAME BRANCH ==='
git -C "$WT" ls-remote --heads personal "refs/heads/$BRANCH"
printf '%s\n' '=== EXISTING PERSONAL COMMIT IDENTITIES ==='
for ref in \
  personal/main \
  personal/codex/idea2-dvac-pi0-robotwin \
  personal/codex/idea2-dvac-train-weighting \
  personal/codex/idea2-dvac-residual-downweight
do
  git -C "$WT" show -s --format="$ref author=%an <%ae> committer=%cn <%ce>" "$ref"
done
printf 'MARKER=SZ_GRPO_COMMIT_GUARD_PROBE_OK\n'
