set -u
v1=/root/autodl-tmp/RLinf_idea2_dvac_train
child=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight

echo '=== configs ==='
find "$v1/examples/embodiment/config" -maxdepth 1 -type f -name '*dvac*formal*.yaml' -printf '%f\n' | sort
find "$child/examples/embodiment/config" -maxdepth 1 -type f -name '*dvac*formal*.yaml' -printf '%f\n' | sort

echo '=== v1 relevant fields ==='
v1_cfg=$(find "$v1/examples/embodiment/config" -maxdepth 1 -type f -name '*dvac*100step*formal*.yaml' | head -n 1)
echo "$v1_cfg"
grep -nE 'log_path|experiment_name|max_steps|dvac|weight|warmup|history|selected_l|stat|clip_z|signal' "$v1_cfg" || true

echo '=== v3 relevant fields ==='
v3_cfg="$child/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal.yaml"
grep -nE 'log_path|experiment_name|max_steps|dvac|weight|warmup|history|selected_l|stat|clip_z|signal' "$v3_cfg" || true

echo '=== schema_and_modes ==='
rg -n 'signal_mode|global_z|global.*z|position_residual|residual|weight_min|weight_max|statistics_mode' \
  "$child/rlinf" "$child/examples/embodiment/config" | head -n 240 || true

echo '=== v1_vs_v3_python_diff_names ==='
git -C "$v1" rev-parse HEAD
git -C "$child" rev-parse HEAD
git -C "$child" status --short
git -C "$child" log --oneline --max-count=6

echo '=== resources_and_targets ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
for p in \
  /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823 \
  /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823; do
  if [ -e "$p" ]; then echo "EXISTS $p"; else echo "ABSENT $p"; fi
done
