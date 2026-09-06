#!/usr/bin/env bash

# Narrow read-only follow-up: exclude static assets and inspect actual eval outputs.
export GIT_OPTIONAL_LOCKS=0

section() {
  printf '\n## %s\n' "$1"
}

section "identity_probe"
date --iso-8601=seconds 2>/dev/null || date
hostname
pwd
id -u

section "actual_norm_stats"
checkpoint=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
find "$checkpoint" -type f -name 'norm_stats.json' \
  -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort

section "rlinf_log_roots"
for logs in /root/autodl-tmp/RLinf*/logs; do
  [[ -d "$logs" ]] || continue
  printf 'logs_root=%s\n' "$logs"
  find "$logs" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -nr | head -n 80
done

section "rlinf_mp4_inventory"
for logs in /root/autodl-tmp/RLinf*/logs; do
  [[ -d "$logs" ]] || continue
  find "$logs" -type f -iname '*.mp4' \
    -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null
done | sort -nr | head -n 160

section "robotwin_rlinf_topology"
for root in /root/autodl-tmp/RoboTwin_RLinf /root/autodl-tmp/RoboTwin; do
  [[ -d "$root" ]] || continue
  printf 'root=%s\n' "$root"
  find "$root" -mindepth 1 -maxdepth 2 \
    -path "$root/.git" -prune -o \
    -path "$root/assets" -prune -o \
    -path "$root/third_party" -prune -o \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %y %p\n' 2>/dev/null | sort -r | head -n 160
done

section "robotwin_nonasset_media"
for root in /root/autodl-tmp/RoboTwin_RLinf /root/autodl-tmp/RoboTwin; do
  [[ -d "$root" ]] || continue
  find "$root" \
    -path "$root/.git" -prune -o \
    -path "$root/assets" -prune -o \
    -path '*/__pycache__' -prune -o \
    -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.gif' \) \
    -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null
done | sort -nr | head -n 200

section "robotwin_recent_nonasset_images_and_results"
root=/root/autodl-tmp/RoboTwin_RLinf
find "$root" \
  -path "$root/.git" -prune -o \
  -path "$root/assets" -prune -o \
  -path '*/__pycache__' -prune -o \
  -type f -newermt '2026-06-12' \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.pkl' -o -iname '*.hdf5' -o -iname '*result*' \) \
  -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -nr | head -n 200

section "vector_env_init_and_save_path"
sed -n '300,365p' /root/autodl-tmp/RoboTwin_RLinf/robotwin/envs/vector_env.py
printf '\nbase_task_init:\n'
sed -n '45,115p' /root/autodl-tmp/RoboTwin_RLinf/envs/_base_task.py
printf '\nbase_task_save_calls:\n'
grep -nE 'save_dir|save_path|save_camera_rgb|merge_pkl|hdf5|eval_video_path|render_freq' \
  /root/autodl-tmp/RoboTwin_RLinf/envs/_base_task.py | head -n 160

section "rlinf_robotwin_constructor_and_cwd"
grep -nE 'RoboTwin|VectorEnv|os.chdir|cwd|save_path|eval_video_log|render_freq|RecordVideo' \
  /root/autodl-tmp/RLinf/rlinf/envs/robotwin/robotwin_env.py | head -n 200
printf '\nconfig_save_switches:\n'
grep -RsnE '^(eval_video_log|render_freq|save_path|save_freq):|eval_video_log:|render_freq:|save_path:' \
  /root/autodl-tmp/RLinf/examples/embodiment/config/env/robotwin.yaml \
  /root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml \
  /root/autodl-tmp/RoboTwin_RLinf/task_config 2>/dev/null | head -n 220

