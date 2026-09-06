#!/usr/bin/env bash
set -euo pipefail
wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
branch=codex/sz-7d07a421-grpo-pi0-robotwin
out="$wt/evidence/completed_runs_20260822_20260903"
test "$(git -C "$wt" rev-parse HEAD)" = 554c6dc8d586162d9444c01fa88308ed4f5203d0
test -d "$out"
test -z "$(find "$out" -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.bin' -o -iname '*.safetensors' -o -iname '*.ckpt' -o -iname '*.mp4' -o -iname '*.npz' -o -iname '*.npy' -o -iname '*.h5' -o -iname '*.hdf5' -o -iname '*.zip' \) -print -quit)"
git -C "$wt" restore --staged -- evidence/completed_runs_20260822_20260903 2>/dev/null || true
find "$out" -type f -name '*.sh' -delete
while IFS= read -r dir; do
  (cd "$dir" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
done < <(find "$out" -mindepth 1 -maxdepth 1 -type d)
git -C "$wt" add -- evidence/completed_runs_20260822_20260903
git -C "$wt" diff --cached --check
git -C "$wt" commit -m 'Add completed Shenzhen GRPO run evidence'
git -C "$wt" push personal "$branch"
head=$(git -C "$wt" rev-parse HEAD)
remote=$(git -C "$wt" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
test "$head" = "$remote"
test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
printf 'PUSHED\t%s\t%s\t%s\t%s\n' "$branch" "$head" "$(find "$out" -type f | wc -l)" "$(du -sb "$out" | awk '{print $1}')"
