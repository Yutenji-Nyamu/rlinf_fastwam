#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_closeout_20260729_v1
expected_head=92e02d9e51c47422696f5ed17a2f15165a6331a6
handoff=HANDOFF.md
ledger=docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
handoff_sha=f4156a8442b09893179268989fa1b9a222ea0a5e28d68a835fab035dba6b4792
ledger_sha=e0461bcf7e6537a03f549b8d7dedd29e364b7fe961f5f016d9c8d7f9492e1cfb

test "$(git -C "${repo}" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'3\t0'
test ! -e /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
test ! -e /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
test "$(sha256sum "${stage}/${handoff}.part" | cut -d' ' -f1)" = \
  "${handoff_sha}"
test "$(sha256sum "${stage}/${ledger}.part" | cut -d' ' -f1)" = \
  "${ledger_sha}"

cp -- "${stage}/${handoff}.part" "${repo}/${handoff}"
cp -- "${stage}/${ledger}.part" "${repo}/${ledger}"
test "$(sha256sum "${repo}/${handoff}" | cut -d' ' -f1)" = "${handoff_sha}"
test "$(sha256sum "${repo}/${ledger}" | cut -d' ' -f1)" = "${ledger_sha}"

expected_status="$(printf '%s\n' "${handoff}" "${ledger}" | LC_ALL=C sort)"
actual_status="$(
  git -C "${repo}" status --porcelain --untracked-files=all \
    | cut -c4- | LC_ALL=C sort
)"
test "${actual_status}" = "${expected_status}"
git -C "${repo}" diff --check

REPO="${repo}" /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import os
from pathlib import Path

repo = Path(os.environ["REPO"])
paths = [
    repo / "HANDOFF.md",
    repo
    / "docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md",
]
for path in paths:
    text = path.read_bytes().decode("utf-8", errors="strict")
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.endswith((" ", "\t")):
            raise ValueError(f"trailing whitespace {path}:{line_number}")
print("STAGE2_CLOSEOUT_CONTENT_QA_OK")
PY

git -C "${repo}" add -- "${handoff}" "${ledger}"
test "$(
  git -C "${repo}" diff --cached --name-only | LC_ALL=C sort
)" = "${expected_status}"
git -C "${repo}" diff --cached --check
git -C "${repo}" commit -m "docs(rlt): record Stage 2 publication state"

test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
printf 'head\t%s\n' "$(git -C "${repo}" rev-parse HEAD)"
printf 'left_right\t%s\n' "$(
  git -C "${repo}" rev-list --left-right --count HEAD...@{upstream}
)"
printf '%s\n' STAGE2_CLOSEOUT_COMMIT_OK
