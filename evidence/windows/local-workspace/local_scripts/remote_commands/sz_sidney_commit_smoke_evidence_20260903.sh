set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
OUT="$WT/evidence/smoke_20260903"
BRANCH=codex/sz-sidney-pi05-current-rlinf
CODE_HEAD=bab221afb8bedc32a8f01b171901a347f0258063

test "$(git -C "$WT" rev-parse HEAD)" = "$CODE_HEAD"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
test -d "$OUT"

STATUS_BEFORE=$(git -C "$WT" status --porcelain --untracked-files=all)
test -n "$STATUS_BEFORE"
printf '%s\n' "$STATUS_BEFORE" | awk '
  substr($0,4) !~ /^evidence\/smoke_20260903\// { bad=1; print "unexpected worktree change: " $0 > "/dev/stderr" }
  END { exit bad }
'

if find "$OUT" -type f -size +1M -print -quit | grep -q .; then
  echo 'unexpected evidence file larger than 1 MiB' >&2
  exit 21
fi

if find "$OUT" -type f \( \
  -iname '*.pt' -o -iname '*.pth' -o -iname '*.bin' -o -iname '*.safetensors' -o \
  -iname '*.ckpt' -o -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o \
  -iname '*.npz' -o -iname '*.npy' \
\) -print -quit | grep -q .; then
  echo 'forbidden model, data, checkpoint, or video payload in evidence' >&2
  exit 22
fi

if grep -ERin --binary-files=without-match \
  -e 'password[[:space:]]*[:=]' \
  -e 'api[_-]\?key[[:space:]]*[:=]' \
  -e 'authorization:[[:space:]]*bearer' \
  -e 'hf_[A-Za-z0-9]\{20,\}' \
  -e 'github_pat_' \
  -e 'BEGIN \(RSA\|OPENSSH\|EC\) PRIVATE KEY' \
  "$OUT"; then
  echo 'credential-like material found in evidence' >&2
  exit 23
fi

(
  cd "$OUT"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
)

BYTES=$(du -sb "$OUT" | awk '{print $1}')
FILES=$(find "$OUT" -type f | wc -l)
test "$BYTES" -lt 1048576

git -C "$WT" add -- evidence/smoke_20260903
STATUS_STAGED=$(git -C "$WT" status --porcelain --untracked-files=all)
printf '%s\n' "$STATUS_STAGED" | awk '
  substr($0,4) !~ /^evidence\/smoke_20260903\// { bad=1; print "unexpected staged/worktree change: " $0 > "/dev/stderr" }
  END { exit bad }
'
git -C "$WT" diff --cached --check
git -C "$WT" commit -m 'Add Sidney pi0.5 smoke evidence'
git -C "$WT" push personal "$BRANCH"

NEW_HEAD=$(git -C "$WT" rev-parse HEAD)
REMOTE_HEAD=$(git -C "$WT" rev-parse "personal/$BRANCH")
test "$NEW_HEAD" = "$REMOTE_HEAD"
test -z "$(git -C "$WT" status --porcelain --untracked-files=all)"

printf 'branch=%s\n' "$BRANCH"
printf 'code_parent=%s\n' "$CODE_HEAD"
printf 'evidence_commit=%s\n' "$NEW_HEAD"
printf 'remote_head=%s\n' "$REMOTE_HEAD"
printf 'evidence_bytes=%s\n' "$BYTES"
printf 'evidence_files=%s\n' "$FILES"
find "$OUT" -type f -printf '%s\t%P\n' | sort -n
