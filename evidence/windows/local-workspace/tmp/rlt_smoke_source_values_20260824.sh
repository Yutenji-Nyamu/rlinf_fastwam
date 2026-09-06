set -u
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac

printf 'BASE_STAGE2_CONFIG_RELEVANT\n'
sed -n '1,360p' "$repo/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml"

printf 'ALL_RLT_SMOKE_CONFIGS\n'
find "$repo/examples/embodiment/config" -maxdepth 1 -type f -iname '*rlt*smoke*.yaml' -print | sort

for f in $(find "$repo/examples/embodiment/config" -maxdepth 1 -type f -iname '*rlt*smoke*.yaml' | sort); do
  printf 'FILE=%s\n' "$f"
  sed -n '1,280p' "$f"
done

printf 'HISTORICAL_RESOURCE_SMOKE_SOURCE_CONFIG\n'
sed -n '1,260p' /root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime/source_config.yaml

printf 'HISTORICAL_RESOURCE_SMOKE_RESOLVED_RELEVANT\n'
grep -nE 'max_steps:|val_check_interval:|save_interval:|total_num_envs:|rollout_epoch:|warmup_min_size:|warmup_post_collect_updates:|max_updates_per_train_step:|warmup_updates:|ramp_updates:|cache_size:|sample_window_size:|experiment_name:|log_path:' /root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime/resolved.yaml

printf 'DVAC_METRIC_AND_STATE_LINES\n'
grep -nE 'rlt_dvac|dvac/|baseline|apply_action_gradient|straight|weight_' "$repo/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" | head -320
