#!/usr/bin/env bash
set -euo pipefail
wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
branch=codex/sz-current-pi0-dvac-grpo-w0to5
out=evidence/completed_runs_20260822_20260903
test "$(git -C "$wt" rev-parse HEAD)" = 0e28ac6f09f821ea12e7d54eba7118ce0000ca86
test -d "$wt/$out"
test -z "$(find "$wt/$out" -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.bin' -o -iname '*.safetensors' \
  -o -iname '*.ckpt' -o -iname '*.mp4' -o -iname '*.npz' -o -iname '*.npy' -o -iname '*.h5' \
  -o -iname '*.hdf5' -o -iname '*.zip' -o -iname '*.tar.gz' \) -print -quit)"
if grep -ERil --binary-files=without-match -e '[REDACTED]' -e 'github_pat_' \
    -e 'hf_[A-Za-z0-9]\{20,\}' -e 'authorization:[[:space:]]*bearer' \
    -e 'BEGIN \(RSA\|OPENSSH\|EC\) PRIVATE KEY' "$wt/$out" | grep -q .; then
  echo credential-like-material >&2; exit 22
fi
git -C "$wt" add -f -- "$out"
git -C "$wt" commit -q -m 'Add completed Shenzhen run evidence'
HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
  timeout 60s git -C "$wt" -c http.version=HTTP/1.1 push -u personal "$branch"
head=$(git -C "$wt" rev-parse HEAD)
remote=$(git -C "$wt" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
test "$head" = "$remote"
test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
printf 'PUSHED\t%s\t%s\t%s\t%s\n' "$branch" "$head" \
  "$(git -C "$wt" ls-files "$out" | wc -l)" "$(du -sb "$wt/$out" | awk '{print $1}')"
