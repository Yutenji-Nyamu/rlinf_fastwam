#!/usr/bin/env bash
set -euo pipefail

echo "=== identity_and_resources ==="
date -Is
hostname
id -u
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
free -h
df -hT /root/autodl-tmp

echo "=== relevant_processes ==="
pgrep -af \
  'ray::|raylet|gcs_server|train_embodied_agent|train_sft|RoboTwin|robotwin|probe_robotwin_rlt' \
  || true

echo "=== worktrees ==="
git -C /root/autodl-tmp/RLinf_rlt_pi0_robotwin \
  status --short --branch
git -C /root/autodl-tmp/RLinf_rlt_pi0_robotwin \
  rev-parse HEAD
git -C /root/autodl-tmp/RLinf_rlt_pi0_robotwin \
  rev-list --left-right --count HEAD...@{u}
git -C /root/autodl-tmp/RLinf_fastwam_rlinf \
  status --short --branch
git -C /root/autodl-tmp/RLinf \
  worktree list --porcelain

echo "=== clean50_existing_targets ==="
for path in \
  /root/autodl-tmp/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1/dataset/adjust_bottle/aloha-agilex_clean_50.zip \
  /root/autodl-tmp/datasets/robotwin2/raw/adjust_bottle/aloha-agilex_clean_50.zip
do
  if [[ -e "$path" ]]; then
    stat --printf='%n|%s|%y\n' "$path"
    sha256sum "$path"
  else
    echo "MISSING|$path"
  fi
done
find /root/autodl-tmp \
  -maxdepth 7 \
  -type f \
  -name 'aloha-agilex_clean_50.zip' \
  -printf '%p|%s|%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' \
  2>/dev/null \
  || true

echo "=== network_capabilities ==="
for name in \
  http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY HF_ENDPOINT
do
  if [[ -n "${!name:-}" ]]; then
    echo "$name=SET"
  else
    echo "$name=UNSET"
  fi
done
command -v hf || true
command -v huggingface-cli || true
/root/autodl-tmp/RLinf/.venv/bin/python -B -c \
  'import huggingface_hub; print("huggingface_hub", huggingface_hub.__version__)'

echo "=== top_level_disk_bytes ==="
du -x -B1 --max-depth=1 /root/autodl-tmp 2>/dev/null \
  | sort -nr \
  | head -30

echo "=== rlinf_fastwam_log_roots ==="
du -x -B1 --max-depth=1 /root/autodl-tmp/RLinf_fastwam_rlinf/logs 2>/dev/null \
  | sort -nr \
  | head -30

echo "=== rlinf_components ==="
du -x -B1 --max-depth=2 /root/autodl-tmp/RLinf 2>/dev/null \
  | sort -nr \
  | head -35

echo "=== robotwin_components ==="
du -x -B1 --max-depth=2 /root/autodl-tmp/RoboTwin 2>/dev/null \
  | sort -nr \
  | head -35

echo "=== old_backup_components ==="
du -x -B1 --max-depth=2 \
  /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 \
  2>/dev/null \
  | sort -nr \
  | head -35

echo "=== model_components ==="
du -x -B1 --max-depth=3 /root/autodl-tmp/models 2>/dev/null \
  | sort -nr \
  | head -40

echo "=== conda_cache_old_components ==="
for root in \
  /root/autodl-tmp/conda \
  /root/autodl-tmp/cache \
  /root/autodl-tmp/RLinf_old_20260618_085536 \
  /root/autodl-tmp/RoboTwin_RLinf \
  /root/autodl-tmp/backups
do
  echo "--- $root"
  du -x -B1 --max-depth=2 "$root" 2>/dev/null \
    | sort -nr \
    | head -20
done

echo "=== dated_large_artifacts ==="
find /root/autodl-tmp \
  -maxdepth 4 \
  -type f \
  -size +5G \
  -printf '%s|%TY-%Tm-%TdT%TH:%TM:%TS%Tz|%p\n' \
  2>/dev/null \
  | sort -nr \
  | head -80
