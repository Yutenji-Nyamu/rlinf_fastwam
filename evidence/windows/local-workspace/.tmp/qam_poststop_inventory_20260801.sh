set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime
run=/root/autodl-tmp/experiments/qam_formal_resume100_to380_20260801_v4/robotwin_adjust_bottle_qam_formal_resume100_to380_20260801_v4
date '+POSTSTOP_TIME=%F %T %Z'
if kill -0 380841 2>/dev/null; then echo DRIVER_ALIVE=yes; else echo DRIVER_ALIVE=no; fi
if kill -0 380842 2>/dev/null; then echo MONITOR_ALIVE=yes; else echo MONITOR_ALIVE=no; fi
pgrep -af 'qam_formal_resume100_to380_20260801_v4|robotwin_adjust_bottle_qam_formal_resume100_to380_20260801_v4' || true
pgrep -af 'train_embodied_agent.py|raylet|gcs_server' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
grep -a 'Global Step:' "$runtime/driver.log" | tail -n 1
find "$run/checkpoints" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V | tail -n 5
du -sh "$runtime" "$run" "$run/checkpoints" 2>/dev/null
find "$runtime" -maxdepth 1 -type f -printf '%f %s\n' | sort
find "$run" -maxdepth 3 -type f \( -name 'complete.json' -o -name 'manifest.json' -o -name '*.yaml' -o -name '*.json' -o -name 'events.out.tfevents*' \) -printf '%p %s\n' | sort
