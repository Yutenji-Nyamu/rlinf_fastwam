set -u
v2=/root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime/driver.log
v3=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime/driver.log

line=$(grep -an -m1 -E 'qam/fine_updates=([1-9]|[0-9]+\.[1-9])' "$v3" | cut -d: -f1)
printf 'first_am_line=%s\n' "$line"
head -n "$line" "$v3" | grep -a 'Global Step:' | tail -1 || true
printf 'v2_committed_steps_1_25\n'
grep -aoE 'success_once=[0-9.]+' "$v2" | head -25 | cut -d= -f2 | \
  awk '{n++;s+=$1} END {printf "cycles=%d episodes=%d successes=%.0f rate=%.6f\n",n,2*n,2*s,s/n}'
printf 'v3_steps_26_51_qonly_candidate\n'
grep -aoE 'success_once=[0-9.]+' "$v3" | head -26 | cut -d= -f2 | \
  awk '{n++;s+=$1} END {printf "cycles=%d episodes=%d successes=%.0f rate=%.6f\n",n,2*n,2*s,s/n}'
printf 'v3_steps_52_100_am_candidate\n'
grep -aoE 'success_once=[0-9.]+' "$v3" | tail -49 | cut -d= -f2 | \
  awk '{n++;s+=$1} END {printf "cycles=%d episodes=%d successes=%.0f rate=%.6f\n",n,2*n,2*s,s/n}'
printf 'full_committed_100\n'
{ grep -aoE 'success_once=[0-9.]+' "$v2" | head -25; grep -aoE 'success_once=[0-9.]+' "$v3"; } | cut -d= -f2 | \
  awk '{n++;s+=$1} END {printf "cycles=%d episodes=%d successes=%.0f rate=%.6f\n",n,2*n,2*s,s/n}'
