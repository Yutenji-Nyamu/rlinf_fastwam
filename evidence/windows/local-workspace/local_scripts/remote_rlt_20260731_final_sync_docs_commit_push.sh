#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/final_sync_docs_staging
branch=codex/rlt-pi0-robotwin
expected_parent=2b8199d8ab2e7b110994fd3234bf7007196c3af9

test "$(git -C "$repo" branch --show-current)" = "$branch"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_parent"
test "$(git -C "$repo" rev-parse personal/$branch)" = "$expected_parent"
test -z "$(git -C "$repo" status --porcelain)"

paths=(
  HANDOFF.md
  docs/rlinf-robotwin-pi0-rltoken/06_AUTODL_NETWORK_PLAYBOOK.md
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
)

for rel in "${paths[@]}"
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
  install -m 0644 "$stage/$rel" "$repo/$rel"
done

mapfile -t changed < <(git -C "$repo" status --porcelain | sed 's/^...//')
test "${#changed[@]}" -eq "${#paths[@]}"
printf '%s\n' "${changed[@]}" | sort > /tmp/rlt_final_sync_changed.txt
printf '%s\n' "${paths[@]}" | sort > /tmp/rlt_final_sync_expected.txt
cmp /tmp/rlt_final_sync_changed.txt /tmp/rlt_final_sync_expected.txt

git -C "$repo" diff --check
git -C "$repo" add -- "${paths[@]}"
git -C "$repo" diff --cached --check
test "$(git -C "$repo" diff --cached --name-only | wc -l)" \
  -eq "${#paths[@]}"

git -C "$repo" commit -m "docs(rlt): record final cloud synchronization"
new_head="$(git -C "$repo" rev-parse HEAD)"
test "$(git -C "$repo" rev-parse HEAD^)" = "$expected_parent"
test -z "$(git -C "$repo" status --porcelain)"

(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u
  remote_head="$(
    timeout 15 git -C "$repo" ls-remote personal \
      "refs/heads/$branch" | cut -f1
  )"
  test "$remote_head" = "$expected_parent"
  GIT_TERMINAL_PROMPT=0 timeout 40 git -C "$repo" push personal \
    "HEAD:refs/heads/$branch"
)

test "$(git -C "$repo" rev-parse personal/$branch)" = "$new_head"
test "$(git -C "$repo" rev-list --left-right --count \
  personal/$branch...HEAD)" = $'0\t0'
test -z "$(git -C "$repo" status --porcelain)"

printf 'FINAL_SYNC_HEAD=%s\n' "$new_head"
git -C "$repo" show --stat --oneline --decorate --no-renames HEAD
