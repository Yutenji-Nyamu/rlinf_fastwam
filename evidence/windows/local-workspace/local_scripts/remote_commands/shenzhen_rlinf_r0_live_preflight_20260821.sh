#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=== IDENTITY / TIME ==='
date --iso-8601=seconds
hostname
pwd
id

printf '%s\n' '=== STORAGE / MEMORY ==='
df -hT / /home /data
df -ih / /home /data
for target in / /home /data; do
  findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS --target "$target"
done
free -h

printf '%s\n' '=== GPU / OWNED PROCESSES ==='
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
ps -u "$(id -u)" -o pid,ppid,stat,etimes,%cpu,%mem,cmd --sort=pid | \
  grep -E 'RLinf|RoboTwin|ray|torchrun|python|uv|huggingface|hf ' || true

printf '%s\n' '=== LOCAL TOOLS ==='
command -v git
git --version
command -v git-lfs || true
git lfs version || true
command -v uv || true
command -v cmake || true
command -v ninja || true
command -v gcc || true
command -v g++ || true
command -v nvcc || true
python3 --version || true

printf '%s\n' '=== TARGET PATHS ==='
for path in \
  /data/chenyiteng/projects/rlinf-shenzhen \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin \
  /data/chenyiteng/models/rlinf \
  /data/chenyiteng/results/rlinf-shenzhen; do
  if [ -e "$path" ]; then
    printf 'EXISTS %s\n' "$path"
    ls -ld "$path"
    du -sh "$path" 2>/dev/null || true
  else
    printf 'ABSENT %s\n' "$path"
  fi
done

printf '%s\n' '=== STANDALONE ASSET SOURCE ==='
standalone=/data/chenyiteng/projects/robotwin-native/RoboTwin
test -d "$standalone/.git"
printf 'robotwin_head='
git -C "$standalone" rev-parse HEAD
printf 'xpolicylab_head='
git -C "$standalone/XPolicyLab" rev-parse HEAD
du -sh "$standalone/assets/background_texture" "$standalone/assets/embodiments" "$standalone/assets/objects"

printf '%s\n' '=== PROXY / OFFICIAL REMOTES ==='
source /etc/profile.d/mihomo-proxy.sh
printf 'https_proxy=%s\n' "${https_proxy:-${HTTPS_PROXY:-unset}}"
printf 'rlinf_main='
git ls-remote https://github.com/RLinf/RLinf.git refs/heads/main | awk '{print $1}'
printf 'robotwin_rlinf_support='
git ls-remote https://github.com/RoboTwin-Platform/RoboTwin.git refs/heads/RLinf_support | awk '{print $1}'
printf 'robotwin_main='
git ls-remote https://github.com/RoboTwin-Platform/RoboTwin.git refs/heads/main | awk '{print $1}'

printf 'github_http='
curl --fail --silent --show-error --location --max-time 30 \
  --output /dev/null --write-out '%{http_code}\n' https://api.github.com/repos/RLinf/RLinf
printf 'pi0_revision_http='
curl --fail --silent --show-error --location --max-time 30 \
  --output /dev/null --write-out '%{http_code}\n' \
  https://huggingface.co/api/models/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/revision/92684e50dca1a5f75adc8d332046c4cf4fa7a3d0
printf 'tokenizer_revision_http='
curl --fail --silent --show-error --location --max-time 30 \
  --output /dev/null --write-out '%{http_code}\n' \
  https://huggingface.co/api/models/RLinf/openpi_tokenizer/revision/befaa248e4f82954b625a421658f933dfd1a97a0
