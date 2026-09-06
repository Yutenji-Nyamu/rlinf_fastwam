set -u
run_root=/root/autodl-tmp/experiments/qam_formal_20260801_v2/robotwin_adjust_bottle_qam_formal_20260801_v2
ckpt="$run_root/checkpoints/global_step_25"

date '+time=%F %T %Z'
hostname
pwd
id -u
printf 'checkpoint=%s\n' "$ckpt"
if test -d "$ckpt"; then
  find "$ckpt" -maxdepth 4 -type f -printf '%P\t%s\n' | sort
  find "$ckpt" -name '.tmp-*' -o -name '*.tmp'
  if test -f "$ckpt/actor/qam_components/complete.json"; then
    printf 'complete_json='
    tr -d '\n' < "$ckpt/actor/qam_components/complete.json"
    printf '\n'
  fi
else
  printf 'checkpoint_missing\n'
fi
printf 'processes\n'
ps -eo pid,ppid,pgid,lstart,cmd | grep -E 'qam_formal_20260801_v2|train_embodied_agent.py|raylet' | grep -v grep || true
printf 'latest_metrics\n'
find "$run_root" -type f \( -name '*.jsonl' -o -name 'metrics*.json' -o -name 'metrics*.log' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5
find "$run_root" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5
