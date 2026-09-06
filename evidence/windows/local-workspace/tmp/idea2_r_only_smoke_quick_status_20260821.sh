#!/usr/bin/env bash
set -u

runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821

for name in wrapper driver observer; do
  pid=$(cat "$runtime/$name.pid" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$name=$pid alive"
  else
    echo "$name=${pid:-absent} exited"
  fi
done

for file in driver.exitcode observer.exitcode launch_finished_at.txt; do
  printf '%s=' "$file"
  cat "$runtime/$file" 2>/dev/null || echo absent
done

grep -aE 'Global Step:|Generating Rollout Epochs:' "$runtime/driver.log" 2>/dev/null | tail -n 14 || true
find "$run/dvac_train" -maxdepth 2 -type f \( -name 'rollout_step0001.npz' -o -name 'runner_step_metrics.csv' \) -printf '%p\n' 2>/dev/null | sort
