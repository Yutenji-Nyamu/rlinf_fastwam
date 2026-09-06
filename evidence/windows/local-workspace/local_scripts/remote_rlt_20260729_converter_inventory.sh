#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=== robotwin_repos ==='
for root in \
  /root/autodl-tmp/RoboTwin \
  /root/autodl-tmp/RoboTwin_RLinf \
  /root/autodl-tmp/RoboTwin2 \
  /root/autodl-tmp/RoboTwin2.0
do
  if [[ -d "$root/.git" ]]; then
    printf 'REPO %s\n' "$root"
    git -C "$root" rev-parse HEAD
    git -C "$root" status --short --branch
  elif [[ -d "$root" ]]; then
    printf 'DIR %s\n' "$root"
  else
    printf 'MISSING %s\n' "$root"
  fi
done

printf '%s\n' '=== converter_candidates ==='
find /root/autodl-tmp/RoboTwin /root/autodl-tmp/RoboTwin_RLinf \
  -maxdepth 7 \
  -type f \
  \( \
    -name 'process_data_pi0.sh' -o \
    -name 'generate.sh' -o \
    -name 'generate.py' -o \
    -iname '*lerobot*.py' -o \
    -iname '*aloha*.py' \
  \) \
  -printf '%s|%TY-%Tm-%TdT%TH:%TM:%TS%Tz|%p\n' \
  2>/dev/null \
  | LC_ALL=C sort

printf '%s\n' '=== exact_known_files ==='
for path in \
  /root/autodl-tmp/RoboTwin/policy/pi0/process_data_pi0.sh \
  /root/autodl-tmp/RoboTwin/policy/pi0/generate.sh \
  /root/autodl-tmp/RoboTwin_RLinf/policy/pi0/process_data_pi0.sh \
  /root/autodl-tmp/RoboTwin_RLinf/policy/pi0/generate.sh
do
  if [[ -f "$path" ]]; then
    sha256sum "$path"
    sed -n '1,240p' "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done

printf '%s\n' '=== stage1_entry_and_checkpoint ==='
worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
find "$worktree" -maxdepth 3 -type f \
  \( -name 'train_sft*.py' -o -name '*sft*.sh' \) \
  -printf '%p\n' \
  | LC_ALL=C sort

model=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
du -sh "$model"
find "$model" -maxdepth 5 -type f \
  \( -name 'norm_stats.json' -o -name '*.json' -o -name '*.safetensors' \) \
  -printf '%s|%TY-%Tm-%TdT%TH:%TM:%TS%Tz|%p\n' \
  | LC_ALL=C sort \
  | head -80
sha256sum "$model/physical-intelligence/robotwin/norm_stats.json"
