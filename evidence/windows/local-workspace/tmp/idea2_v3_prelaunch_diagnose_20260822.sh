#!/usr/bin/env bash
source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
config="$source_root/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal.yaml"
printf 'HEAD=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
printf 'STATUS_BEGIN\n'; git -C "$source_root" status --short; printf 'STATUS_END\n'
printf 'RUN_EXISTS=%s\n' "$(test -e "$run_dir" && echo 1 || echo 0)"
printf 'RESOLVED_EXISTS=%s\n' "$(test -f "$runtime_dir/resolved_config.yaml" && echo 1 || echo 0)"
sha256sum "$runtime_dir/resolved_config.yaml" "$config" "$runtime_dir/launch_formal.sh" "$runtime_dir/observe_resources.sh"
printf 'KEYS\n'; grep -nE 'max_steps:|weight_min:|weight_max:|warmup_steps:|history_window_steps:|selected_l:' "$config"
printf 'MATCHING_PROCESSES\n'; pgrep -af 'train_embodied_agent.py.*robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal' || true
bash -n "$runtime_dir/launch_formal.sh"; printf 'LAUNCH_SYNTAX_RC=%s\n' "$?"
bash -n "$runtime_dir/observe_resources.sh"; printf 'OBSERVER_SYNTAX_RC=%s\n' "$?"
