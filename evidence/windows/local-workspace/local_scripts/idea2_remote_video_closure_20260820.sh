#!/usr/bin/env bash

export GIT_OPTIONAL_LOCKS=0

section() {
  printf '\n## %s\n' "$1"
}

section "identity_probe"
date --iso-8601=seconds 2>/dev/null || date
hostname
pwd
id -u

section "norm_stats_lock"
norm=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
stat -c '%s bytes %y %n' "$norm"
sha256sum "$norm"

section "exact_robotwin_roots"
for path in \
  /root/autodl-tmp/RoboTwin_RLinf \
  /root/autodl-tmp/RoboTwin_RLinf/data \
  /root/autodl-tmp/RoboTwin_RLinf/eval_result \
  /root/autodl-tmp/RoboTwin \
  /root/autodl-tmp/RoboTwin/data \
  /root/autodl-tmp/RoboTwin/eval_result; do
  if [[ -e "$path" ]]; then
    stat -c '%F %y %n' "$path"
  else
    printf 'ABSENT %s\n' "$path"
  fi
done

section "robotwin_rlinf_nonasset_mp4_count_and_newest"
root=/root/autodl-tmp/RoboTwin_RLinf
count=$(find "$root" -path "$root/.git" -prune -o -path "$root/assets" -prune -o -type f -iname '*.mp4' -print 2>/dev/null | wc -l)
printf 'count=%s\n' "$count"
find "$root" -path "$root/.git" -prune -o -path "$root/assets" -prune -o -type f -iname '*.mp4' \
  -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -nr | head -n 20

section "standalone_robotwin_mp4_count_and_newest"
root=/root/autodl-tmp/RoboTwin
count=$(find "$root/eval_result" -type f -iname '*.mp4' -print 2>/dev/null | wc -l)
printf 'count=%s\n' "$count"
find "$root/eval_result" -type f -iname '*.mp4' \
  -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -nr | head -n 10

section "relative_task_reward_directories"
find /root/autodl-tmp -mindepth 2 -maxdepth 5 -type d -name 'adjust_bottle_reward' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort

section "rlinf_eval_run_precise"
run=/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40/logs/logs_his/20260618-11:24:19-robotwin_adjust_bottle_openpi_eval_autodl
stat -c '%F %y %n' "$run"
find "$run/video/eval" -type f -iname '*.mp4' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | head -n 4
find "$run/video/eval" -type f -iname '*.mp4' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 4

section "rlinf_robotwin_constructor"
sed -n '55,105p' /root/autodl-tmp/RLinf/rlinf/envs/robotwin/robotwin_env.py
printf '\nrun_eval_entry:\n'
sed -n '1,95p' /root/autodl-tmp/RLinf/evaluations/run_eval.sh

