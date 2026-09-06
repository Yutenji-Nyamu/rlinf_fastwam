set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
base="$repo/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml"
dvac="$repo/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2.yaml"
printf 'BASE_CONFIG\n'
cat "$base"
printf 'DVAC_CONFIG\n'
cat "$dvac"
printf 'HISTORICAL_LAUNCH_CONFIGS\n'
find /root/autodl-tmp -maxdepth 4 -type f \( -iname '*resume*480*.yaml' -o -iname '*resume*480*.sh' -o -iname '*250*480*.yaml' -o -iname '*250*480*.sh' \) -printf '%p\n' | sort
printf 'HISTORICAL_COMMAND_REFERENCES\n'
rg -n 'max_steps=480|max_steps: 480|resume_dir|resume250|250_to480|250-to-480' \
  /root/autodl-tmp/rlt* /root/autodl-tmp/experiment_exports /root/autodl-tmp/RLinf_rlt_pi0_robotwin 2>/dev/null | head -n 400 || true
