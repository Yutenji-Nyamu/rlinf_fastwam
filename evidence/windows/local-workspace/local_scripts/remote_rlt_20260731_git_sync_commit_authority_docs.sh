#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/authority_docs_staging
expected_head=d81267b57eb5ce13e6452139aaeba02af3911624

test "$(git -C "$repo" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain)"

for rel in \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
do
  test -f "$stage/$rel"
  python - "$stage/$rel" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
raw = path.read_bytes()
assert not raw.startswith(b"\xef\xbb\xbf"), f"BOM: {path}"
text = raw.decode("utf-8")
assert text.endswith("\n"), f"missing final newline: {path}"
assert not re.search(r"[ \t]+$", text, re.MULTILINE), f"trailing whitespace: {path}"
assert len(re.findall(r"(?m)^```", text)) % 2 == 0, f"odd fence count: {path}"
PY
done

install -m 0644 "$stage/HANDOFF.md" "$repo/HANDOFF.md"
install -m 0644 \
  "$stage/docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md" \
  "$repo/docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md"

mapfile -t changed < <(git -C "$repo" status --porcelain | sed 's/^...//')
test "${#changed[@]}" -eq 2
printf '%s\n' "${changed[@]}" | sort > /tmp/rlt_authority_changed.txt
printf '%s\n' \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md \
  | sort > /tmp/rlt_authority_expected.txt
cmp /tmp/rlt_authority_changed.txt /tmp/rlt_authority_expected.txt

git -C "$repo" diff --check
git -C "$repo" add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
git -C "$repo" diff --cached --check
test "$(git -C "$repo" diff --cached --name-only | wc -l)" -eq 2

git -C "$repo" commit -m "docs(rlt): clarify cross-machine Git authority"

printf 'HEAD=%s\n' "$(git -C "$repo" rev-parse HEAD)"
printf 'PARENT=%s\n' "$(git -C "$repo" rev-parse HEAD^)"
printf 'STATUS_BEGIN\n'
git -C "$repo" status --short --branch
printf 'STATUS_END\n'
git -C "$repo" show --stat --oneline --decorate --no-renames HEAD
