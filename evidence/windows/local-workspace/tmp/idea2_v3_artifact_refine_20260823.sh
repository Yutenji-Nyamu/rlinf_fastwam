set -u
RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822

echo '=== CHECKPOINT_PATHS ==='
find "$RUN" -maxdepth 5 -type d -name 'global_step_*' -print | sort -V
while IFS= read -r d; do
  [ -n "$d" ] || continue
  du -sh "$d"
  find "$d" -maxdepth 3 -type f -name 'complete.json' -print -exec cat {} \;
done < <(find "$RUN" -maxdepth 5 -type d -name 'global_step_*' -print | sort -V)

echo '=== LATEST_NPZ ==='
for rank in 00 01; do
  find "$RUN/dvac_train/actor_rank${rank}" -maxdepth 1 -type f -name 'rollout_step*.npz' -printf '%f %s\n' | sort -V | tail -n 3
done

echo '=== TRACEBACK_CONTEXT ==='
grep -ain -B 3 -A 7 'Traceback' "$RUNTIME/driver.log" || true

echo '=== WRAPPER_OBSERVER_TAIL ==='
tail -n 20 "$RUNTIME/wrapper.log" 2>/dev/null || true
tail -n 20 "$RUNTIME/observer.log" 2>/dev/null || true

echo '=== LATEST_FILE_TIMES ==='
stat -c '%y %s %n' \
  "$RUN/metrics.log" \
  "$RUN/dvac_train/actor_rank00/runner_step_metrics.csv" \
  "$RUN/dvac_train/actor_rank01/runner_step_metrics.csv" \
  "$RUNTIME/driver.log" \
  "$RUNTIME/resource_monitor/resources.csv"
