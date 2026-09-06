#!/usr/bin/env bash
set -euo pipefail

source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
config="$source_root/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal.yaml"
expected_head=eb2a09176c362c7386895ca4f3680b92aeb0ee5b
expected_resolved_sha=bbe3db1f6184778765301b284c15319f83f73f4e2e4b1a509c0f1c01648eadd4

printf 'IDENTITY\n'
hostname
pwd
id -u

test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$source_root" status --short)"
test ! -e "$run_dir"
test -f "$runtime_dir/resolved_config.yaml"
test "$(sha256sum "$runtime_dir/resolved_config.yaml" | awk '{print $1}')" = "$expected_resolved_sha"
bash -n "$runtime_dir/launch_formal.sh"
bash -n "$runtime_dir/observe_resources.sh"
test "$(grep -Ec '^    weight_min: 0.0$' "$config")" -eq 1
test "$(grep -Ec '^    weight_max: 2.0$' "$config")" -eq 1
test "$(grep -Ec '^  max_steps: 100$' "$config")" -eq 1
test "$(pgrep -af '[t]rain_embodied_agent.py.*robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal' | wc -l)" -eq 0

printf 'SOURCE_HEAD=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
printf 'SOURCE_STATUS=%s\n' "$(git -C "$source_root" status --short)"
printf 'CONFIG_SHA256=%s\n' "$(sha256sum "$config" | awk '{print $1}')"
printf 'RESOLVED_SHA256=%s\n' "$(sha256sum "$runtime_dir/resolved_config.yaml" | awk '{print $1}')"
printf 'SCRIPT_SHA256\n'
sha256sum "$runtime_dir/launch_formal.sh" "$runtime_dir/observe_resources.sh"
printf 'CONFIG_KEYS\n'
grep -E '^(  max_steps:|    weight_min:|    weight_max:|    warmup_steps:|    history_window_steps:|    selected_l:)' "$config"
printf 'GPU_STATE\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'MEMORY_CURRENT_BYTES=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'MEMORY_EVENTS\n'
cat /sys/fs/cgroup/memory.events
printf 'DISK\n'
df -h /root/autodl-tmp /dev/shm
printf 'PRELAUNCH_AUDIT_OK\n'
