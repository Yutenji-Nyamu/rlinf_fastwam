#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

TOKEN_REV=befaa248e4f82954b625a421658f933dfd1a97a0
TOKEN_SHA=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
TOKEN_SIZE=4264023
TARGET=/home/chenyiteng/.cache/openpi
PARTIAL=/home/chenyiteng/.cache/.partial-openpi-tokenizer-befaa248-20260821-retry1
HF_PYTHON=/home/chenyiteng/miniforge3/envs/RoboTwin/bin/python

test ! -e "$TARGET"
test ! -e "$PARTIAL"
test -x "$HF_PYTHON"

mkdir -p "$PARTIAL"
timeout --signal=INT --kill-after=30s 600s \
  "$HF_PYTHON" - "$TOKEN_REV" "$PARTIAL" <<'PY'
from huggingface_hub import snapshot_download
import sys

print(snapshot_download(
    repo_id="RLinf/openpi_tokenizer",
    revision=sys.argv[1],
    local_dir=sys.argv[2],
    allow_patterns=["big_vision/paligemma_tokenizer.model"],
    max_workers=1,
))
PY

MODEL="$PARTIAL/big_vision/paligemma_tokenizer.model"
test "$(stat -c %s "$MODEL")" = "$TOKEN_SIZE"
printf '%s  %s\n' "$TOKEN_SHA" "$MODEL" | sha256sum --check -

# RLinf 7d07's official helper checks a root-level path although OpenPI reads
# big_vision/paligemma_tokenizer.model. Point that check at the same pinned file.
ln -s big_vision/paligemma_tokenizer.model "$PARTIAL/paligemma_tokenizer.model"
test "$(readlink -f "$PARTIAL/paligemma_tokenizer.model")" = "$(readlink -f "$MODEL")"

mv "$PARTIAL" "$TARGET"
test "$(stat -c %s "$TARGET/big_vision/paligemma_tokenizer.model")" = "$TOKEN_SIZE"
printf '%s  %s\n' "$TOKEN_SHA" "$TARGET/big_vision/paligemma_tokenizer.model" | sha256sum --check -

printf '%s\n' '=== PINNED TOKENIZER MANIFEST ==='
find "$TARGET" -maxdepth 4 -printf '%y %s %p -> %l\n' | sort
du -sh "$TARGET"
printf 'tokenizer_revision=%s\n' "$TOKEN_REV"
printf '%s\n' 'R1_PINNED_TOKENIZER_OK'
