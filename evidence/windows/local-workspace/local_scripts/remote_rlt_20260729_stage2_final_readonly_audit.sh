#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
code_commit=3b610cb4685a1d41c97da64df67ab86561697dfd
expected_head=4f3062762043558a22c375eb415e636e08de9369
stage1_endpoint=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_acceptance=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2
pre_smoke=/root/autodl-tmp/experiment_exports/rlt_stage2_pre_smoke_20260729_v1
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
smoke_evidence=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
fresh=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh
resume=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_resume.sh
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh

test "$(git -C "${repo}" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'4\t0'
git -C "${repo}" merge-base --is-ancestor "${code_commit}" HEAD
git -C "${repo}" diff --quiet \
  "${code_commit}" -- . ':(exclude)docs/**' ':(exclude)HANDOFF.md'
test ! -e "${smoke_root}"
test ! -e "${smoke_evidence}"
test -f "${stage1_endpoint}/actor/model_state_dict/full_weights.pt"
test -f "${stage1_acceptance}/stage1_artifact_manifest.json"
test -f "${stage1_acceptance}/validation.json"
test -f "${pre_smoke}/resolved_contract_audit.json"

printf 'audit_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "$(git -C "${repo}" branch --show-current)"
printf 'head\t%s\n' "$(git -C "${repo}" rev-parse HEAD)"
printf 'left_right\t%s\n' "$(
  git -C "${repo}" rev-list --left-right --count HEAD...@{upstream}
)"
printf 'status\tCLEAN\n'
git -C "${repo}" log -4 --format='commit	%H	%s'

printf '%s\n' process_audit
processes="$(
  pgrep -af \
    'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2|git .*push.*codex/rlt-pi0-robotwin' \
    | grep -v -E 'pgrep -af|stage2_final_readonly_audit' || true
)"
if test -n "${processes}"; then
  printf '%s\n' "${processes}"
  exit 1
fi
printf '%s\n' NONE

nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
awk '
  /MemTotal:/ {total=$2}
  /MemAvailable:/ {available=$2}
  END {
    printf "host_total_kib\t%s\nhost_available_kib\t%s\n", total, available
  }
' /proc/meminfo
printf 'cgroup_current_bytes\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
awk '
  $1 == "anon" {printf "cgroup_anon_bytes\t%s\n", $2}
  $1 == "file" {printf "cgroup_file_bytes\t%s\n", $2}
' /sys/fs/cgroup/memory.stat
awk '{printf "cgroup_event_%s\t%s\n", $1, $2}' \
  /sys/fs/cgroup/memory.events
df -B1 --output=size,used,avail,pcent /root/autodl-tmp | tail -n 1 \
  | awk '{
      printf "disk_total_bytes\t%s\n", $1
      printf "disk_used_bytes\t%s\n", $2
      printf "disk_available_bytes\t%s\n", $3
      printf "disk_percent\t%s\n", $4
    }'

sha256sum \
  "${stage1_acceptance}/stage1_artifact_manifest.json" \
  "${stage1_acceptance}/validation.json" \
  "${pre_smoke}/formal_bound_resolved.yaml" \
  "${pre_smoke}/fresh_bound_resolved.yaml" \
  "${pre_smoke}/resume_bound_resolved.yaml" \
  "${pre_smoke}/resolved_contract_audit.json" \
  "${fresh}" "${resume}" "${monitor}"

ROOT="${repo}" ACCEPTANCE="${stage1_acceptance}" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
import os
from pathlib import Path

acceptance = Path(os.environ["ACCEPTANCE"])
validation = json.loads((acceptance / "validation.json").read_text())
print(f"stage1_accepted\t{validation['accepted']}")
print(f"stage1_all_gates\t{all(validation['gates'].values())}")
print(
    "stage1_losses\t"
    f"fresh={validation['metrics']['fresh_seed0_proxy_loss']:.10f},"
    f"true={validation['metrics']['true_z_loss']:.10f},"
    f"shuffled={validation['metrics']['shuffled_z_loss']:.10f},"
    f"zero={validation['metrics']['zero_z_loss']:.10f}"
)
print(
    "stage1_non_rlt_changed\t"
    f"{validation['reload_contract']['non_rlt_changed_tensor_count']}"
)
PY

printf 'stage1_endpoint_bytes\t%s\n' "$(
  du -sb "${stage1_endpoint}" | cut -f1
)"
printf 'smoke_root_exists\t%s\n' "$(
  test -e "${smoke_root}" && printf yes || printf no
)"
printf 'smoke_evidence_exists\t%s\n' "$(
  test -e "${smoke_evidence}" && printf yes || printf no
)"
printf '%s\n' STAGE2_FINAL_READONLY_AUDIT_OK
