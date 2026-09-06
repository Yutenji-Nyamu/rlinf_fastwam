set -u
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac

printf 'DVAC_OVERLAY\n'
cfg="$repo/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2.yaml"
sed -n '1,280p' "$cfg"

printf 'DVAC_COMMIT_DIFFSTAT\n'
git -C "$repo" show --stat --oneline --decorate --no-renames HEAD

printf 'DVAC_CONFIG_KEYS_IN_SOURCE\n'
grep -RInE 'teacher_dvac|warmup_min_size|baseline_ready|runner_step_metrics|raw_trace' \
  "$repo/rlinf/algorithms/rlt" \
  "$repo/rlinf/workers" \
  "$repo/rlinf/data" \
  "$repo/rlinf/models/embodiment/openpi" \
  "$cfg" 2>/dev/null | head -320

printf 'HISTORICAL_8ENV3C_SMOKE_CONFIG\n'
find "$repo" -type f -name '*8env3c*.yaml' -print -exec sed -n '1,260p' {} \;

printf 'HISTORICAL_FRESH_SMOKE_LAUNCH\n'
sed -n '1,300p' /root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh

printf 'HISTORICAL_RESOURCE_SMOKE_EXPORT\n'
find /root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1 -maxdepth 3 -type f -printf '%s %p\n' | sort -n | head -120

printf 'HISTORICAL_RESOURCE_SMOKE_LOG_TAIL\n'
for f in $(find /root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1 -type f \( -name '*.log' -o -name '*.txt' \) | sort); do
  printf 'FILE=%s\n' "$f"
  tail -80 "$f"
done

printf 'STAGE1_CHECKPOINT_SIZE\n'
du -sh /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000

printf 'BASE_RUNTIME_ENV_HINTS\n'
find /root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1 -maxdepth 3 -type f -printf '%s %p\n' | sort -n | head -120
