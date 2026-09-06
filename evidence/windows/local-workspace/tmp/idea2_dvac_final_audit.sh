set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1
output="$target/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1"
test "$(cat "$runtime_dir/driver_exit_code.txt")" = 0
grep -q '^STATUS=COMPLETE$' "$runtime_dir/status.env"
test "$(git -C "$target" rev-parse HEAD)" = 61996e15cc7f5a32bd6012b61b20893d94636c82
test -z "$(git -C "$target" status --porcelain)"
test "$(ps -eo args= | grep -F "$target" | grep -v grep | wc -l)" = 0
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)" = 0
test "$(find "$output/dvac_telemetry" -type f -name 'trace_rollout_rank*.npz' | wc -l)" = 2
test "$(find "$output/dvac_telemetry" -type f -name 'query_index_rollout_rank*.csv' | wc -l)" = 2
test "$(find "$output/dvac_telemetry" -type f -name 'episode_index_env_rank*.csv' | wc -l)" = 2
test "$(find "$output/dvac_telemetry/query_images" -type f -name '*.png' | wc -l)" = 192
test "$(find "$output/video/eval" -type f -name '*.mp4' | wc -l)" = 2
test -s "$runtime_dir/resource_summary.json"
printf 'FINAL_AUDIT_AT=%s\n' "$(date --iso-8601=seconds)"
printf 'DRIVER_EXIT_CODE=0\nSTATUS=COMPLETE\nSOURCE_CLEAN=1\n'
printf 'MATCHING_PROCESSES=0\nGPU_COMPUTE_PROCESSES=0\n'
printf 'TRACE_SHARDS=2\nQUERY_SHARDS=2\nEPISODE_SHARDS=2\nQUERY_IMAGES=192\nMP4=2\n'
printf 'MEMORY_EVENTS=%s\n' "$(tr '\n' ';' < /sys/fs/cgroup/memory.events)"
printf 'FINAL_AUDIT_PASS=1\n'
