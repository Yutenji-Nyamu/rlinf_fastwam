set -u
v3=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime/driver.log
for n in 10 20 30 40 49; do
  printf 'last_%s_am_cycles\t' "$n"
  grep -aoE 'success_once=[0-9.]+' "$v3" | tail -"$n" | cut -d= -f2 | \
    awk '{c++;s+=$1} END {printf "episodes=%d successes=%.0f rate=%.4f\n",2*c,2*s,s/c}'
done
