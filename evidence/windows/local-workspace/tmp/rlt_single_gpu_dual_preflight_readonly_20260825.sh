#!/usr/bin/env bash
set -u

echo '[IDENTITY]'
date -Is
hostname
pwd
id -u

echo '[PROCESSES]'
ps -eo pid,ppid,pgid,lstart,etimes,rss,cmd --sort=pid | grep -E 'rlt_teacher_dvac|train_embodied_agent|ray::|RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker' | grep -v grep || true

echo '[OLD_RUN_LIFECYCLE]'
old_run=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
old_runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
for f in started_at finished_at exit_code foreground.log driver.log resources.csv resolved.yaml source_commit.txt config_sha256.txt; do
  p="$old_runtime/$f"
  if [ -e "$p" ]; then
    printf '%s\t' "$p"
    if [ -f "$p" ] && [ "$(stat -c %s "$p")" -le 4096 ]; then
      tr '\n' ' ' < "$p"
      echo
    else
      stat -c 'size=%s mtime=%y' "$p"
    fi
  fi
done
if [ -f "$old_run/metrics.log" ]; then
  echo '[OLD_RUN_LAST_STEPS]'
  grep -E 'Global Step|success_once|eval/success_once|rlt_dvac/weight_mean|actor_grad_norm' "$old_run/metrics.log" | tail -n 80 || true
fi
find "$old_run/checkpoints" -maxdepth 2 -type f \( -name 'checkpoint_complete.json' -o -name 'rlt_trainer_state_complete.json' \) -printf '%p\n' 2>/dev/null | sort -V | tail -n 20 || true
du -sh "$old_run" "$old_runtime" 2>/dev/null || true

echo '[GPU]'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

echo '[CGROUP]'
for f in memory.current memory.high memory.max memory.swap.max memory.events; do
  p="/sys/fs/cgroup/$f"
  if [ -r "$p" ]; then
    echo "-- $f"
    cat "$p"
  fi
done

echo '[DISK]'
df -h /root/autodl-tmp

echo '[WORKTREE]'
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac rev-parse HEAD
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac status --short
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac branch --show-current

echo '[CONFIG_FILES]'
ls -l /root/autodl-tmp/RLinf_rlt_teacher_dvac/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp*.yaml

