set -u
v2=/root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime/driver.log
v3=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime/driver.log

printf 'first_nonzero_fine_context\n'
grep -a -B 16 -m 1 -E 'qam/fine_updates=([1-9]|[0-9]+\.[1-9])' "$v3" 2>/dev/null || true
printf 'last_steps_context\n'
grep -a -B 16 'qam/fine_updates=' "$v3" 2>/dev/null | tail -100 || true
printf 'success_counts_v2_v3\n'
for file in "$v2" "$v3"; do
  printf '%s\t' "$file"
  grep -aoE 'success_once=[0-9.]+' "$file" 2>/dev/null | cut -d= -f2 | \
    awk '{n+=1; s+=$1} END {printf "cycles=%d success_sum=%.6f episodes=%d successes=%.0f rate=%.6f\n", n,s,2*n,2*s,(n? s/n:0)}'
done
printf 'step_times_v3\n'
grep -aoE 'Step Time: [0-9.]+s' "$v3" 2>/dev/null | tail -10 || true
printf 'runtime_markers\n'
grep -aE 'Resuming training|Saving checkpoint|Global Step:  *100/100|QAM_FORMAL_EXIT' "$v3" 2>/dev/null | tail -20 || true
