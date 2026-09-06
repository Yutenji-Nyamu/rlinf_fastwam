set -u
child=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
cfg="$child/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal.yaml"
echo '=== global_z_config_block ==='
sed -n '55,90p' "$cfg"
echo '=== residual_config_block ==='
sed -n '55,82p' "$child/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal.yaml"
echo '=== mode_implementation ==='
grep -RInE 'signal_mode|global_all_queries_all_h|per_h_robust_residual|weight_min|weight_max|clip_z|strength' \
  "$child/rlinf/algorithms" "$child/rlinf/workers" 2>/dev/null | head -n 220 || true
echo '=== config_diff_v1_vs_v3_excluding_paths ==='
diff -u "$cfg" "$child/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal.yaml" | sed -n '1,240p' || true
