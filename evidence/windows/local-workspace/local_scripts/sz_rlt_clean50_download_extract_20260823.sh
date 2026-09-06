#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REV=9dc9299c163db059931898a9f0852098a61155a1
ROOT=/data/chenyiteng/datasets/robotwin2
SOURCE=$ROOT/source/$REV
ZIP=$SOURCE/dataset/adjust_bottle/aloha-agilex_clean_50.zip
RAW_PARENT=$ROOT/raw/$REV/adjust_bottle
RAW=$RAW_PARENT/clean50-v1
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

mkdir -p "$SOURCE" "$RAW_PARENT" /data/chenyiteng/cache/huggingface
test ! -e "$RAW"

export HF_HOME=/data/chenyiteng/cache/huggingface
export HF_HUB_DISABLE_XET=1
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

echo "START $(date --iso-8601=seconds)"
echo "TARGET_ZIP=$ZIP"
echo "TARGET_RAW=$RAW"
df -h /data

"$PY" -B - <<'PY'
from huggingface_hub import hf_hub_download

print(hf_hub_download(
    repo_id="TianxingChen/RoboTwin2.0",
    repo_type="dataset",
    revision="9dc9299c163db059931898a9f0852098a61155a1",
    filename="dataset/adjust_bottle/aloha-agilex_clean_50.zip",
    local_dir="/data/chenyiteng/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1",
))
PY

test "$(stat -c '%s' "$ZIP")" -eq 298659710
test "$(sha256sum "$ZIP" | awk '{print $1}')" = 5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e
unzip -tqq "$ZIP"

STAGING=$(mktemp -d "$RAW_PARENT/.clean50-v1.extract.XXXXXX")
unzip -q "$ZIP" -d "$STAGING"
EXTRACTED=$STAGING/aloha-agilex_clean_50
test "$(find "$EXTRACTED/data" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)" -eq 50
test "$(find "$EXTRACTED/instructions" -maxdepth 1 -type f -name 'episode*.json' | wc -l)" -eq 50
mv -T "$EXTRACTED" "$RAW"
rmdir "$STAGING"

echo "RAW_EPISODES=$(find "$RAW/data" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)"
du -sh "$ZIP" "$RAW"
echo "SUCCESS $(date --iso-8601=seconds)"
