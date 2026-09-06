#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_terminal_ledger_20260729_v1
expected_head=4f3062762043558a22c375eb415e636e08de9369
ledger=docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
ledger_sha=6d0aa96cfa6910cdfea2f8a4a1837e27768cdd03e457cebeced0c50d2a3088c7
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
smoke_evidence=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1

test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'4\t0'
test "$(sha256sum "${stage}/IMPLEMENTATION_LOG.md.part" | cut -d' ' -f1)" = \
  "${ledger_sha}"
test ! -e "${smoke_root}"
test ! -e "${smoke_evidence}"

cp -- "${stage}/IMPLEMENTATION_LOG.md.part" "${repo}/${ledger}"
test "$(sha256sum "${repo}/${ledger}" | cut -d' ' -f1)" = "${ledger_sha}"
test "$(
  git -C "${repo}" status --porcelain --untracked-files=all | cut -c4-
)" = "${ledger}"
git -C "${repo}" diff --check
git -C "${repo}" add -- "${ledger}"
test "$(git -C "${repo}" diff --cached --name-only)" = "${ledger}"
git -C "${repo}" diff --cached --check
git -C "${repo}" commit -m "docs(rlt): close Stage 2 pre-smoke ledger"

test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'5\t0'
test ! -e "${smoke_root}"
test ! -e "${smoke_evidence}"
if pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2|git .*push.*codex/rlt-pi0-robotwin' \
  | grep -v -E 'pgrep -af|stage2_terminal_ledger_commit' >/dev/null
then
  printf '%s\n' "Unexpected RLT/Ray/push process after terminal commit." >&2
  exit 1
fi

printf 'head\t%s\n' "$(git -C "${repo}" rev-parse HEAD)"
printf 'left_right\t%s\n' "$(
  git -C "${repo}" rev-list --left-right --count HEAD...@{upstream}
)"
nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
printf '%s\n' STAGE2_TERMINAL_LEDGER_COMMIT_OK
