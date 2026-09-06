#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
resolved_sha=77419258766880fca7b8d15d725dfb9f2fba5b88d2bfe21eb918fd760e9a7bc9
source_sha=f777a0caceacf260c427093fdb0a493f6bd77c3d444cf233a57727ce9027c291

declare -A expected=(
  [remote_ogpo_20260807_formal_prepare_v1.sh]=008dca1ee58140a759c64e8bd1a22ff73b359b5c63f35bc0400e2c69b4cfeb22
  [remote_ogpo_20260807_formal_prepare_resume_v2.sh]=44ab3b63ec3c030050daef153bfa34098cdb47b6e72cdd317474c3536edaee02
  [remote_ogpo_20260807_formal_partial_inspect_v1.sh]=0420b8a5235337e4b9093111f96f1961bf096a7c31e3aca1879bac678d30aaa6
  [remote_ogpo_20260807_formal_run_v1.sh]=c61a712d1aca5d3a717b462f869d4c8689e42406fbe505d6f5a59c3afbffe0a2
  [remote_ogpo_20260807_formal_monitor_v1.sh]=e7c16a18561224c3915686291d48ffc9cbaf795d0fa6e259f7913a2303e9dad5
  [remote_ogpo_20260807_formal_status_v1.sh]=739c7c5b6d236487e0dab6cf603fb60e93648417d9845fb3cab1a441532d684f
)

cd "$repo"
test "$(git branch --show-current)" = codex/ogpo-pi0-robotwin
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --short)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum "$runtime_root/resolved.yaml" | awk '{print $1}')" = "$resolved_sha"
test "$(sha256sum "$runtime_root/source_config.yaml" | awk '{print $1}')" = "$source_sha"
test -s "$runtime_root/exact_command.txt"
test -s "$runtime_root/run_provenance.tsv"
test -s "$runtime_root/stop_conditions.txt"
test -s "$runtime_root/resources_before.txt"
test ! -e "$run_root"
test ! -e "$runtime_root/started_at.txt"

for name in "${!expected[@]}"; do
  path="$runtime_root/$name"
  test -f "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "${expected[$name]}"
  bash -n "$path"
done

run_script="$runtime_root/remote_ogpo_20260807_formal_run_v1.sh"
grep -Fq 'algorithm.ogpo.total_online_rows=35000' "$run_script"
grep -Fq 'algorithm.ogpo.start_training_rows=10000' "$run_script"
grep -Fq 'algorithm.ogpo.utd_q=0.1' "$run_script"
grep -Fq 'algorithm.ogpo.utd_pi=0.1' "$run_script"
grep -Fq 'algorithm.ogpo.replay_capacity=40000' "$run_script"
if grep -Fq 'timeout --signal' "$run_script"; then
  printf '%s\n' 'unexpected smoke wall timeout in formal runner' >&2
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
printf 'RESOLVED_SHA256\t%s\n' "$resolved_sha"
printf 'EXACT_COMMAND_SHA256\t%s\n' "$(sha256sum "$runtime_root/exact_command.txt" | awk '{print $1}')"
printf 'PROVENANCE_SHA256\t%s\n' "$(sha256sum "$runtime_root/run_provenance.tsv" | awk '{print $1}')"
printf 'STOP_CONDITIONS_SHA256\t%s\n' "$(sha256sum "$runtime_root/stop_conditions.txt" | awk '{print $1}')"
for name in $(printf '%s\n' "${!expected[@]}" | sort); do
  printf 'SCRIPT_SHA256\t%s\t%s\n' "${expected[$name]}" "$name"
done
printf 'EXACT_COMMAND\t%s\n' "$(cat "$runtime_root/exact_command.txt")"
printf '%s\n' OGPO_ROBOTWIN_FORMAL_RUNTIME_VERIFIED
