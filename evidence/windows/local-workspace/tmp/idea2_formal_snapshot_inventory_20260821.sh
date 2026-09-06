run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
rt=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821
find "$run/control_trace" -type f -printf '%p %s\n' 2>/dev/null
stat -c '%n %s' \
  "$rt/resource_monitor/resources.csv" \
  "$rt/resource_monitor/process_rss.tsv" \
  "$rt/driver.log"
find "$run/dvac_train" -type f -name '*.npz' -printf '%s\n' | \
  awk '{s += $1; n += 1} END {print "npz_count=" n, "npz_bytes=" s}'
