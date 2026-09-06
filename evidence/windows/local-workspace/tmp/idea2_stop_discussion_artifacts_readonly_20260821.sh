#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821

date --iso-8601=seconds
printf 'run_top_level\n'
find "$run" -mindepth 1 -maxdepth 1 -printf '%y %s %f\n' | sort
printf 'run_top_level_sizes\n'
du -sh "$run"/* 2>/dev/null | sort -h
printf 'checkpoint_candidates\n'
find "$run" -maxdepth 3 \( -type d -o -type f \) \( -iname '*global_step*' -o -iname '*checkpoint*' -o -iname '*metadata*' \) -printf '%y %s %p\n' 2>/dev/null | sort | tail -n 120
printf 'runtime_inventory\n'
find "$runtime" -maxdepth 2 -type f -printf '%s %p\n' | sort -n
printf 'wrapper_script\n'
sed -n '1,260p' "$runtime/run_formal_100step.sh"
printf 'current_process_group\n'
ps -g 114145 -o pid=,ppid=,pgid=,sid=,stat=,etimes=,cmd= | head -n 120
