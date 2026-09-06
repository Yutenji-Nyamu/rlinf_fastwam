#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
fresh=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh

printf 'audit_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "$(git -C "${repo}" branch --show-current)"
printf 'head\t%s\n' "$(git -C "${repo}" rev-parse HEAD)"
printf 'left_right\t%s\n' "$(
  git -C "${repo}" rev-list --left-right --count HEAD...@{upstream}
)"
printf 'upstream\t%s\n' "$(git -C "${repo}" rev-parse '@{upstream}')"
printf 'status_begin\n'
git -C "${repo}" status --short --branch
printf 'status_end\n'

personal_url="$(git -C "${repo}" remote get-url personal)"
personal_url="$(
  printf '%s\n' "${personal_url}" \
    | sed -E 's#(https?://)[^/@]+@#\1<redacted>@#'
)"
printf 'personal_url\t%s\n' "${personal_url}"
printf 'http_version\t%s\n' "$(
  git -C "${repo}" config --get http.version || printf DEFAULT
)"
printf 'credential_helper\t%s\n' "$(
  git -C "${repo}" config --get credential.helper || printf NONE
)"
for name in http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY; do
  if test -n "${!name-}"; then
    printf 'env_%s\tSET\n' "${name}"
  else
    printf 'env_%s\tUNSET\n' "${name}"
  fi
done

probe_url() {
  label=$1
  url=$2
  result="$(
    timeout 10s curl -L -sS -o /dev/null \
      --connect-timeout 7 --max-time 9 \
      -w '%{http_code},%{remote_ip},%{time_connect},%{time_total}' \
      "${url}"
  )" || result="ERROR"
  printf 'curl_%s\t%s\n' "${label}" "${result}"
}
probe_url github https://github.com
probe_url api https://api.github.com
probe_url raw https://raw.githubusercontent.com

remote_head="$(
  timeout 15s git -C "${repo}" ls-remote \
    personal refs/heads/codex/rlt-pi0-robotwin 2>/dev/null \
    | cut -f1
)" || remote_head=ERROR
printf 'ls_remote\t%s\n' "${remote_head:-EMPTY}"

printf 'processes_begin\n'
pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2|git .*push.*codex/rlt-pi0-robotwin' \
  | grep -v -E 'pgrep -af|fresh_smoke_live_audit' \
  || printf '%s\n' NONE
printf 'processes_end\n'

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
printf 'smoke_root_exists\t%s\n' "$(
  test -e "${smoke_root}" && printf yes || printf no
)"
printf 'evidence_root_exists\t%s\n' "$(
  test -e "${evidence_root}" && printf yes || printf no
)"

sha256sum \
  "${repo}/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml" \
  "${repo}/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml" \
  "${repo}/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
  "${manifest}" \
  "${fresh}" \
  "${monitor}"
printf '%s\n' RLT_FRESH_SMOKE_LIVE_AUDIT_OK
