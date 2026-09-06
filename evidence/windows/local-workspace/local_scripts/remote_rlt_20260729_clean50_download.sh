#!/usr/bin/env bash
set -euo pipefail

revision=9dc9299c163db059931898a9f0852098a61155a1
repo_id=TianxingChen/RoboTwin2.0
filename=dataset/adjust_bottle/aloha-agilex_clean_50.zip
expected_size=298659710
expected_sha256=5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e
local_root=/root/autodl-tmp/datasets/robotwin2/source/${revision}
target=${local_root}/${filename}
lock_path=/root/autodl-tmp/tmp/rlt_clean50_download.lock

mkdir -p "$(dirname "$target")" /root/autodl-tmp/cache/huggingface
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "FAIL: another clean-50 download owns $lock_path" >&2
  exit 20
fi

echo "START $(date -Is)"
echo "repo=${repo_id}"
echo "revision=${revision}"
echo "filename=${filename}"
echo "target=${target}"
echo "expected_size=${expected_size}"
echo "expected_sha256=${expected_sha256}"
df -h /root/autodl-tmp

if [[ -e "$target" ]]; then
  actual_size=$(stat -c '%s' "$target")
  actual_sha256=$(sha256sum "$target" | awk '{print $1}')
  if [[ "$actual_size" == "$expected_size" \
        && "$actual_sha256" == "$expected_sha256" ]]; then
    unzip -tqq "$target"
    echo "ALREADY_VALID"
    echo "SUCCESS $(date -Is)"
    exit 0
  fi
  echo "FAIL: target exists but does not match the source lock" >&2
  echo "actual_size=${actual_size}" >&2
  echo "actual_sha256=${actual_sha256}" >&2
  exit 21
fi

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1

echo "network_path=hf-mirror.com"
echo "hf_home=${HF_HOME}"

/root/autodl-tmp/RLinf/.venv/bin/python -B -c \
  "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='${repo_id}', filename='${filename}', repo_type='dataset', revision='${revision}', local_dir='${local_root}'))"

actual_size=$(stat -c '%s' "$target")
actual_sha256=$(sha256sum "$target" | awk '{print $1}')
echo "actual_size=${actual_size}"
echo "actual_sha256=${actual_sha256}"

if [[ "$actual_size" != "$expected_size" ]]; then
  echo "FAIL: size mismatch" >&2
  exit 22
fi
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "FAIL: sha256 mismatch" >&2
  exit 23
fi

unzip -t "$target"
echo "archive_entries=$(unzip -Z1 "$target" | wc -l)"
echo "archive_uncompressed_bytes=$(unzip -l "$target" | awk 'END {print $1}')"
echo "archive_top_level_sample"
unzip -Z1 "$target" | awk 'NR <= 30 {print}'
df -h /root/autodl-tmp
echo "SUCCESS $(date -Is)"
