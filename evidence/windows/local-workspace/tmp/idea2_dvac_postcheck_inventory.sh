set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1
output="$target/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1"

printf 'POSTCHECK_AT=%s\n' "$(date --iso-8601=seconds)"
printf 'SOURCE_HEAD=%s\n' "$(git -C "$target" rev-parse HEAD)"
printf 'SOURCE_DIRTY_COUNT=%s\n' "$(git -C "$target" status --porcelain | wc -l)"
printf 'REMOTE_HEAD=%s\n' "$(git -C "$target" rev-parse personal/codex/idea2-dvac-pi0-robotwin)"
printf '%s\n' STATUS_BEGIN
cat "$runtime_dir/status.env"
printf '%s\n' STATUS_END
printf 'DRIVER_EXIT_CODE_FILE=%s\n' "$(cat "$runtime_dir/driver_exit_code.txt")"
printf 'MATCHING_PROCESS_COUNT=%s\n' "$(ps -eo args= | grep -F "$target" | grep -v grep | wc -l)"
printf 'GPU_COMPUTE_PROCESS_COUNT=%s\n' "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
printf '%s\n' GPU_BEGIN
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' GPU_END
printf 'MEMORY_CURRENT=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'MEMORY_EVENTS=%s\n' "$(tr '\n' ';' < /sys/fs/cgroup/memory.events)"
printf 'OUTPUT_BYTES=%s\n' "$(du -sb "$output" | cut -f1)"
printf 'OUTPUT_FILES=%s\n' "$(find "$output" -type f | wc -l)"
printf '%s\n' TOP_LEVEL_BEGIN
find "$output" -maxdepth 3 -type f -printf '%P\t%s\n' | sort
printf '%s\n' TOP_LEVEL_END
printf 'NPZ_COUNT=%s\n' "$(find "$output" -type f -name 'trace_rollout_rank*.npz' | wc -l)"
printf 'QUERY_CSV_COUNT=%s\n' "$(find "$output" -type f -name 'query_index_rollout_rank*.csv' | wc -l)"
printf 'EPISODE_CSV_COUNT=%s\n' "$(find "$output" -type f -name 'episode_index_env_rank*.csv' | wc -l)"
printf 'QUERY_PNG_COUNT=%s\n' "$(find "$output" -type f -path '*/query_images/*' -name '*.png' | wc -l)"
printf 'MP4_COUNT=%s\n' "$(find "$output" -type f -name '*.mp4' | wc -l)"
printf '%s\n' MP4_LIST_BEGIN
find "$output" -type f -name '*.mp4' -printf '%P\t%s\n' | sort
printf '%s\n' MP4_LIST_END
printf '%s\n' ROBOTWIN_NATIVE_MP4_DURING_RUN_BEGIN
find /root/autodl-tmp/RoboTwin_RLinf -type f -name '*.mp4' -newermt '2026-08-20 19:03:17' ! -newermt '2026-08-20 19:07:12' -printf '%p\t%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort
printf '%s\n' ROBOTWIN_NATIVE_MP4_DURING_RUN_END
