#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
resolved_sha=352f8e80752d60624a0c53c62d21dcc10bdc8e6712a433c18d0eec0ee1f56a36
source_sha=f777a0caceacf260c427093fdb0a493f6bd77c3d444cf233a57727ce9027c291
run_sha=ed694eeadae6e6e8506bee45aeb500a20f2a9402f2db7eab60c4ba12aa556ee3
monitor_sha=505665dab6f3afc2082b156e7c6296a4f4002a1132e566624a886e500acdc0a1
health_sha=67b497778fb0718918c17e4392ebe0b58afabdb049c3b38d349a62c788c890ff
run_script="$runtime_root/remote_ogpo_20260808_formal_run_v2.sh"
monitor_script="$runtime_root/remote_ogpo_20260808_formal_monitor_v2.sh"
health_script="$runtime_root/remote_ogpo_20260808_formal_health_v2.sh"

cd "$repo"
test "$(git branch --show-current)" = codex/ogpo-pi0-robotwin
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --short)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum "$runtime_root/resolved.yaml" | awk '{print $1}')" = "$resolved_sha"
test "$(sha256sum "$runtime_root/source_config.yaml" | awk '{print $1}')" = "$source_sha"
test "$(sha256sum "$run_script" | awk '{print $1}')" = "$run_sha"
test "$(sha256sum "$monitor_script" | awk '{print $1}')" = "$monitor_sha"
test "$(sha256sum "$health_script" | awk '{print $1}')" = "$health_sha"
test -s "$runtime_root/exact_command.txt"
test -s "$runtime_root/run_provenance.tsv"
test -s "$runtime_root/stop_conditions.txt"
test -s "$runtime_root/resources_before.txt"
test ! -e "$run_root"
test ! -e "$runtime_root/launched_at.txt"
test ! -e "$runtime_root/started_at.txt"

bash -n "$run_script"
bash -n "$monitor_script"
bash -n "$health_script"

for fragment in \
  'algorithm.ogpo.total_online_rows=90000' \
  'algorithm.ogpo.start_training_rows=10000' \
  'algorithm.ogpo.utd_q=0.05' \
  'algorithm.ogpo.utd_pi=0.05' \
  'algorithm.ogpo.replay_capacity=100000' \
  'algorithm.ogpo.baseline_eval=true' \
  'algorithm.ogpo.final_eval=true' \
  'algorithm.ogpo.eval_interval_rows=10000' \
  'algorithm.ogpo.checkpoint_interval_rows=30000'; do
  grep -Fq "$fragment" "$run_script"
  grep -Fq "$fragment" "$runtime_root/exact_command.txt"
done
if grep -Fq 'timeout --signal' "$run_script"; then
  printf '%s\n' 'unexpected wall timeout in formal runner' >&2
  exit 8
fi

mapfile -t active_rows < <(
  {
    ps -eo pid=,comm=,args= \
      | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
test "${#active_rows[@]}" = 0
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF')"

printf 'VERIFIED_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'HEAD\t%s\n' "$expected_head"
for name in resolved.yaml source_config.yaml exact_command.txt run_provenance.tsv \
  stop_conditions.txt resources_before.txt \
  remote_ogpo_20260808_formal_run_v2.sh \
  remote_ogpo_20260808_formal_monitor_v2.sh \
  remote_ogpo_20260808_formal_health_v2.sh; do
  printf 'SHA256\t%s\t%s\n' "$(sha256sum "$runtime_root/$name" | awk '{print $1}')" "$name"
done
printf 'EXACT_COMMAND\t%s\n' "$(cat "$runtime_root/exact_command.txt")"
printf '%s\n' OGPO_ROBOTWIN_FORMAL_V2_RUNTIME_VERIFIED
