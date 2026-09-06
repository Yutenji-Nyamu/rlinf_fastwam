#!/usr/bin/env bash
set -euo pipefail

PIN=7faa71108368fbb3b6885649f112af607427a2d4
REPO_URL=https://github.com/yuantianyuan01/FastWAM.git
SOURCE_ROOT=/data/chenyiteng/projects/fastwam-standalone
FW="$SOURCE_ROOT/FastWAM-7faa711"
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
MODEL_ROOT=/data/chenyiteng/models/fastwam
RESULT_ROOT=/data/chenyiteng/results/fastwam-standalone
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
hostname
id

printf '%s\n' '=== exact targets before write ==='
for target in "$FW" "$ENV" "$MODEL_ROOT" "$RESULT_ROOT" "$CACHE_ROOT"; do
  if [[ -e "$target" || -L "$target" ]]; then
    printf 'PRESENT %s\n' "$target"
  else
    printf 'ABSENT %s\n' "$target"
  fi
done

test ! -e "$FW"
test ! -L "$FW"
test ! -e "$ENV"
test ! -L "$ENV"

printf '%s\n' '=== capacity and target GPU ==='
df -hT /home /data
df -ih /home /data
nvidia-smi -i 0 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

printf '%s\n' '=== network and remote source lock ==='
source /etc/profile.d/mihomo-proxy.sh
printf 'https_proxy=%s\n' "${https_proxy:-unset}"
REMOTE_HEAD="$(git ls-remote "$REPO_URL" refs/heads/main | awk '{print $1}')"
printf 'remote_main=%s\n' "$REMOTE_HEAD"
test "$REMOTE_HEAD" = "$PIN"

printf '%s\n' '=== clone and detach exact official source ==='
mkdir -p "$SOURCE_ROOT"
git clone "$REPO_URL" "$FW"
git -C "$FW" checkout --detach "$PIN"
test "$(git -C "$FW" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$FW" status --porcelain)"

printf '%s\n' '=== source manifest ==='
git -C "$FW" remote -v
git -C "$FW" show -s --format='commit=%H%ncommit_time=%cI%nsubject=%s' HEAD
git -C "$FW" status --short
du -sh "$FW"
printf '%s\n' FASTWAM_R0_SOURCE_OK
