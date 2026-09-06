set -u

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
printf 'RLT_REPO\n'
if [ ! -d "$repo/.git" ] && [ ! -f "$repo/.git" ]; then
  printf 'missing %s\n' "$repo"
  exit 0
fi
git -C "$repo" status --short --branch
git -C "$repo" rev-parse HEAD
git -C "$repo" branch --show-current
git -C "$repo" remote -v

printf 'RLT_CONFIGS\n'
find "$repo/examples" -type f -name '*rlt*robotwin*.yaml' -o -name '*robotwin*rlt*.yaml' | sort

printf 'RLT_DVAC_EXISTING_SYMBOLS\n'
grep -R -n -E 'dvac|endpoint|z_endpoint|gradient_weight|pi_for_q|actor_loss =|qf_pi =|bc_loss' \
  "$repo/rlinf/algorithms/rlt" \
  "$repo/rlinf/models/embodiment/openpi" \
  "$repo/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
  "$repo/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py" 2>/dev/null | head -n 240 || true

printf 'RLT_RUNS\n'
find /root/autodl-tmp/experiments -maxdepth 3 -type d \( -name 'global_step_250' -o -name 'global_step_480' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -n 12
