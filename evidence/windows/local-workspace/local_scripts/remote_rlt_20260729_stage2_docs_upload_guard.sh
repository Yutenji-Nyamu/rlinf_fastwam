#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
expected_head=3b610cb4685a1d41c97da64df67ab86561697dfd
failed_stage=/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v1
stage=/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v2
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
smoke_evidence=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1

test "$(git -C "${repo}" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_head}"
test -f "${failed_stage}/UPLOAD_SHA256SUMS"
test ! -e "${stage}"
test ! -e "${smoke_root}"
test ! -e "${smoke_evidence}"

failed_expected="$(
  sed 's/^[0-9a-f]\{64\}  //' \
    "${failed_stage}/UPLOAD_SHA256SUMS" | LC_ALL=C sort
)"
actual_dirty="$(
  git -C "${repo}" status --porcelain --untracked-files=all \
    | cut -c4- | LC_ALL=C sort
)"
test -n "${actual_dirty}"
unexpected="$(
  comm -23 \
    <(printf '%s\n' "${actual_dirty}") \
    <(printf '%s\n' "${failed_expected}")
)"
test -z "${unexpected}"

if pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2' \
  | grep -v -E 'pgrep -af|stage2_docs_upload_guard' >/dev/null
then
  printf '%s\n' "Refusing docs upload while an RLT/Ray process is active." >&2
  exit 1
fi

printf 'timestamp\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "$(git -C "${repo}" branch --show-current)"
printf 'head\t%s\n' "$(git -C "${repo}" rev-parse HEAD)"
printf 'upstream_left_right\t%s\n' \
  "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})"
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
awk '/MemAvailable:/ {printf "host_available_kib\t%s\n", $2}' /proc/meminfo
df -B1 --output=avail /root/autodl-tmp | tail -n 1 \
  | awk '{printf "autodl_tmp_available_bytes\t%s\n", $1}'
printf '%s\n' "STAGE2_DOCS_UPLOAD_GUARD_OK"
