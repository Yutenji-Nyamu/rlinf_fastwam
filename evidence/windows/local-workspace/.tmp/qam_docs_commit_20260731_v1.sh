#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
stage=/root/autodl-tmp/qam_formal_docs_sync_20260731_v1
expected_head=4a15699e10971e306ed756dcbbf8aa65632553d5
ledger=docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
printf '%s  %s\n' \
  cee47c18d4bf0ee89d5a7bd20a3a2f5267102268ed654e10f7c4ede32e1e250f \
  "$stage/$ledger" | sha256sum -c -
install -D -m 0644 "$stage/$ledger" "$repo/$ledger"

files=(
  HANDOFF.md
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_code_commit_push_20260731.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_launch_20260731_v1.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_patch_apply_20260731.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resolved_20260731_v1.yaml
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_schedule_server_tests_20260731.sh
  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_resolved_20260731_formal_v1.yaml
  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_to_formal_20260731_v1.diff
)

diff -u \
  <(printf '%s\n' "${files[@]}" | sort) \
  <(git -C "$repo" status --porcelain=v1 | sed 's/^...//' | sort)

git -C "$repo" diff --check -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_code_commit_push_20260731.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_launch_20260731_v1.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_patch_apply_20260731.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resolved_20260731_v1.yaml \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_schedule_server_tests_20260731.sh \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_source_resolved_20260731_formal_v1.yaml

if grep -R -n -F 'NBo7SQqoatnZ' "${files[@]/#/$repo/}"; then
  printf 'SECRET_SCAN_FAIL=password\n' >&2
  exit 1
fi

git -C "$repo" add -- "${files[@]}"
git -C "$repo" diff --cached --check
diff -u \
  <(printf '%s\n' "${files[@]}" | sort) \
  <(git -C "$repo" diff --cached --name-only | sort)
git -C "$repo" diff --cached --stat
git -C "$repo" commit -m 'docs(qam): record formal launch'
printf 'DOCS_COMMIT='
git -C "$repo" rev-parse HEAD
printf 'TREE='
git -C "$repo" rev-parse 'HEAD^{tree}'
git -C "$repo" status --short
