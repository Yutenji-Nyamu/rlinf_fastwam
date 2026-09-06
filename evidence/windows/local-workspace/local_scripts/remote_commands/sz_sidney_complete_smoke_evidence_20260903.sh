set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
OUT="$WT/evidence/smoke_20260903"
BRANCH=codex/sz-sidney-pi05-current-rlinf
FIRST_EVIDENCE_COMMIT=e8998bdc23c8d2c5f75e904e66aa34b1eeb34f09

test "$(git -C "$WT" rev-parse HEAD)" = "$FIRST_EVIDENCE_COMMIT"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
STATUS_BEFORE=$(git -C "$WT" status --porcelain --untracked-files=all)
printf '%s\n' "$STATUS_BEFORE" | awk '
  NF && substr($0,4) !~ /^evidence\/smoke_20260903\// { bad=1; print "unexpected staged/worktree change: " $0 > "/dev/stderr" }
  END { exit bad }
'

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

# Mechanical whitespace cleanup for the copied human-readable excerpts only.
sed -i 's/[[:space:]]\+$//' "$OUT/grpo_step1/command.txt"
perl -0pi -e 's/\n+\z/\n/' "$OUT/b1_eval/metrics.log" "$OUT/grpo_step1/metrics.log"
(
  cd "$OUT"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
)

git -C "$WT" add -f -- evidence/smoke_20260903
STATUS_STAGED=$(git -C "$WT" status --porcelain --untracked-files=all)
test -n "$STATUS_STAGED"
printf '%s\n' "$STATUS_STAGED" | awk '
  substr($0,4) !~ /^evidence\/smoke_20260903\// { bad=1; print "unexpected staged/worktree change: " $0 > "/dev/stderr" }
  END { exit bad }
'

DISK_FILES=$(find "$OUT" -type f | wc -l)
INDEX_FILES=$(git -C "$WT" ls-files evidence/smoke_20260903 | wc -l)
test "$DISK_FILES" = 37
test "$INDEX_FILES" = "$DISK_FILES"
git -C "$WT" diff --cached --check
git -C "$WT" commit -m 'Complete Sidney pi0.5 smoke evidence bundle'
git -C "$WT" push personal "$BRANCH"

NEW_HEAD=$(git -C "$WT" rev-parse HEAD)
REMOTE_HEAD=$(git -C "$WT" rev-parse "personal/$BRANCH")
test "$NEW_HEAD" = "$REMOTE_HEAD"
test -z "$(git -C "$WT" status --porcelain --untracked-files=all)"

BYTES=$(du -sb "$OUT" | awk '{print $1}')
TRACKED_BYTES=$(git -C "$WT" ls-files -z evidence/smoke_20260903 | xargs -0 -I{} stat -c %s "$WT/{}" | awk '{sum += $1} END {print sum}')
test "$BYTES" = "$TRACKED_BYTES"

printf 'branch=%s\n' "$BRANCH"
printf 'first_evidence_commit=%s\n' "$FIRST_EVIDENCE_COMMIT"
printf 'final_evidence_commit=%s\n' "$NEW_HEAD"
printf 'remote_head=%s\n' "$REMOTE_HEAD"
printf 'evidence_bytes=%s\n' "$BYTES"
printf 'evidence_files=%s\n' "$DISK_FILES"
printf 'tracked_bytes=%s\n' "$TRACKED_BYTES"
git -C "$WT" show --stat --oneline --summary HEAD
