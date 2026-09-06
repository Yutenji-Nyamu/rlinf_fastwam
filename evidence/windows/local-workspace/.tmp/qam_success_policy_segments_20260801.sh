set -u
v2=/root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime/driver.log
v3=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime/driver.log

printf 'base_policy_steps_1_52\t'
{ grep -aoE 'success_once=[0-9.]+' "$v2" | head -25; grep -aoE 'success_once=[0-9.]+' "$v3" | head -27; } | cut -d= -f2 | \
  awk '{n++;s+=$1} END {printf "episodes=%d successes=%.0f rate=%.4f\n",2*n,2*s,s/n}'
printf 'updated_fine_steps_53_100\t'
grep -aoE 'success_once=[0-9.]+' "$v3" | tail -48 | cut -d= -f2 | \
  awk '{n++;s+=$1} END {printf "episodes=%d successes=%.0f rate=%.4f\n",2*n,2*s,s/n}'
printf 'step26_collect\t'
grep -aoE 'success_once=[0-9.]+' "$v3" | head -1
printf 'step52_pre_am_rollout\t'
grep -aoE 'success_once=[0-9.]+' "$v3" | head -27 | tail -1
