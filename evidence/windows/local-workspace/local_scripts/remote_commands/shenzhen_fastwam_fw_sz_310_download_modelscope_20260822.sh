#!/usr/bin/env bash
set -euo pipefail

# FW-SZ-310: only the T5, VAE, and tokenizer required by Fast-WAM RoboTwin
# release inference. The 5B DiT is intentionally excluded because the official
# sim config initializes it from the released Fast-WAM checkpoint.
# Quota is root-only. Run shenzhen_mihomo_quota_sanitized_20260821.sh
# immediately before and after this command and retain all three outputs together.

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
FW_PIN=7faa71108368fbb3b6885649f112af607427a2d4
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
BASE=/data/chenyiteng/models/fastwam/diffsynth
# ModelScope 1.34.0 documents commit hashes but supports branch/tag resolution
# in snapshot_download. Therefore each master ref is checked against the exact
# audited commit immediately before the selective download; final payload
# hashes are the second, race-safe acceptance gate.
CONVERTED_REPO=DiffSynth-Studio/Wan-Series-Converted-Safetensors
CONVERTED_REVISION=master
CONVERTED_COMMIT=150f75d811d51f6c7760154aa7fec371dccda529
TOKENIZER_REPO=Wan-AI/Wan2.1-T2V-1.3B
TOKENIZER_REVISION=master
TOKENIZER_COMMIT=020b9e59db399efa534e408fb295b14a75700daf
CONVERTED_ROOT="$BASE/$CONVERTED_REPO"
TOKENIZER_ROOT="$BASE/$TOKENIZER_REPO"
T5_REL=models_t5_umt5-xxl-enc-bf16.safetensors
VAE_REL=Wan2.2_VAE.safetensors
SPECIAL_REL=google/umt5-xxl/special_tokens_map.json
SPIECE_REL=google/umt5-xxl/spiece.model
TOKENIZER_JSON_REL=google/umt5-xxl/tokenizer.json
TOKENIZER_CONFIG_REL=google/umt5-xxl/tokenizer_config.json
T5_BYTES=11361845432
T5_SHA256=d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23
VAE_BYTES=1409401152
VAE_SHA256=0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996
SPECIAL_BYTES=6623
SPECIAL_SHA256=7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d
SPIECE_BYTES=4548313
SPIECE_SHA256=e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458
TOKENIZER_JSON_BYTES=16837417
TOKENIZER_JSON_SHA256=6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b
TOKENIZER_CONFIG_BYTES=61728
TOKENIZER_CONFIG_SHA256=ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05
EXPECTED_PAYLOAD_BYTES=12792700665
DISK_MARGIN_BYTES=2147483648
PYTHON="$ENV/bin/python"

tree_bytes() {
  if [[ -d "$1" ]]; then
    du -sb -- "$1" | awk '{print $1}'
  else
    printf '0\n'
  fi
}

check_existing_or_absent() {
  local path=$1 expected_bytes=$2 expected_sha=$3
  if [[ -e "$path" || -L "$path" ]]; then
    if [[ ! -f "$path" || -L "$path" ]]; then
      printf 'REFUSE_NONREGULAR_TARGET=%s\n' "$path" >&2
      exit 50
    fi
    local actual_bytes actual_sha
    actual_bytes=$(stat -c '%s' -- "$path")
    actual_sha=$(sha256sum -- "$path" | awk '{print $1}')
    if [[ "$actual_bytes" != "$expected_bytes" || "$actual_sha" != "$expected_sha" ]]; then
      printf 'REFUSE_MISMATCHED_EXISTING_FILE=%s bytes=%s sha256=%s\n' \
        "$path" "$actual_bytes" "$actual_sha" >&2
      exit 51
    fi
    printf 'existing_complete=%s bytes=%s sha256=%s\n' \
      "$path" "$actual_bytes" "$actual_sha"
  else
    printf 'existing_absent=%s\n' "$path"
  fi
}

printf 'fw_stage=FW-SZ-310\ntimestamp_start=%s\n' "$(date --iso-8601=seconds)"
printf 'converted_repo=%s\nconverted_revision=%s\nconverted_commit=%s\n' \
  "$CONVERTED_REPO" "$CONVERTED_REVISION" "$CONVERTED_COMMIT"
printf 'tokenizer_repo=%s\ntokenizer_revision=%s\ntokenizer_commit=%s\nbase=%s\nexpected_payload_bytes=%s\n' \
  "$TOKENIZER_REPO" "$TOKENIZER_REVISION" "$TOKENIZER_COMMIT" "$BASE" "$EXPECTED_PAYLOAD_BYTES"
printf '%s\n' \
  'quota_before_contract=run sanitized root-only quota probe immediately before this command' \
  'quota_after_contract=run sanitized root-only quota probe immediately after this command'

test -x "$PYTHON"
test "$(git -C "$FW" rev-parse HEAD)" = "$FW_PIN"
"$PYTHON" - <<'PY'
from importlib.metadata import version

assert version("modelscope") == "1.34.0", version("modelscope")
print("modelscope=" + version("modelscope"))
PY

printf '%s\n' '=== before: exact target files and capacity ==='
base_bytes_before=$(tree_bytes "$BASE")
data_free_before=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
home_free_before=$(df -B1 --output=avail /home | tail -n 1 | tr -d '[:space:]')
printf 'base_bytes_before=%s\ndata_free_bytes_before=%s\nhome_free_bytes_before=%s\n' \
  "$base_bytes_before" "$data_free_before" "$home_free_before"
check_existing_or_absent "$CONVERTED_ROOT/$T5_REL" "$T5_BYTES" "$T5_SHA256"
check_existing_or_absent "$CONVERTED_ROOT/$VAE_REL" "$VAE_BYTES" "$VAE_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$SPECIAL_REL" "$SPECIAL_BYTES" "$SPECIAL_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$SPIECE_REL" "$SPIECE_BYTES" "$SPIECE_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$TOKENIZER_JSON_REL" "$TOKENIZER_JSON_BYTES" "$TOKENIZER_JSON_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$TOKENIZER_CONFIG_REL" "$TOKENIZER_CONFIG_BYTES" "$TOKENIZER_CONFIG_SHA256"

missing_bytes=0
[[ -f "$CONVERTED_ROOT/$T5_REL" ]] || missing_bytes=$((missing_bytes + T5_BYTES))
[[ -f "$CONVERTED_ROOT/$VAE_REL" ]] || missing_bytes=$((missing_bytes + VAE_BYTES))
[[ -f "$TOKENIZER_ROOT/$SPECIAL_REL" ]] || missing_bytes=$((missing_bytes + SPECIAL_BYTES))
[[ -f "$TOKENIZER_ROOT/$SPIECE_REL" ]] || missing_bytes=$((missing_bytes + SPIECE_BYTES))
[[ -f "$TOKENIZER_ROOT/$TOKENIZER_JSON_REL" ]] || missing_bytes=$((missing_bytes + TOKENIZER_JSON_BYTES))
[[ -f "$TOKENIZER_ROOT/$TOKENIZER_CONFIG_REL" ]] || missing_bytes=$((missing_bytes + TOKENIZER_CONFIG_BYTES))
required_data_free=$((missing_bytes + DISK_MARGIN_BYTES))
printf 'missing_payload_bytes=%s\nrequired_data_free_with_2gib_margin=%s\n' \
  "$missing_bytes" "$required_data_free"
if (( data_free_before < required_data_free )); then
  printf 'INSUFFICIENT_DATA_FREE=%s required=%s\n' "$data_free_before" "$required_data_free" >&2
  exit 52
fi

source /etc/profile.d/mihomo-proxy.sh
mkdir -p \
  "$CONVERTED_ROOT" \
  "$TOKENIZER_ROOT/google/umt5-xxl" \
  "$CACHE_ROOT/modelscope" \
  "$CACHE_ROOT/tmp"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH="$BASE"
export TMPDIR="$CACHE_ROOT/tmp"
printf 'MODELSCOPE_CACHE=%s\nDIFFSYNTH_DOWNLOAD_SOURCE=%s\nDIFFSYNTH_MODEL_BASE_PATH=%s\nTMPDIR=%s\n' \
  "$MODELSCOPE_CACHE" "$DIFFSYNTH_DOWNLOAD_SOURCE" "$DIFFSYNTH_MODEL_BASE_PATH" "$TMPDIR"
env | grep -iE '^(http|https|all|no)_proxy=' \
  | sed -E 's#(https?://)[^/@]+@#\1REDACTED@#I' \
  | sort
curl -sSIL --max-time 30 --connect-timeout 10 -o /dev/null \
  -w 'modelscope_converted_probe http=%{http_code} remote=%{remote_ip} total=%{time_total}\n' \
  "https://www.modelscope.cn/models/$CONVERTED_REPO"
curl -sSIL --max-time 30 --connect-timeout 10 -o /dev/null \
  -w 'modelscope_tokenizer_probe http=%{http_code} remote=%{remote_ip} total=%{time_total}\n' \
  "https://www.modelscope.cn/models/$TOKENIZER_REPO"
converted_remote_commit=$( \
  git ls-remote "https://www.modelscope.cn/$CONVERTED_REPO.git" refs/heads/master \
  | awk '{print $1}' \
)
tokenizer_remote_commit=$( \
  git ls-remote "https://www.modelscope.cn/$TOKENIZER_REPO.git" refs/heads/master \
  | awk '{print $1}' \
)
printf 'converted_remote_master=%s\ntokenizer_remote_master=%s\n' \
  "$converted_remote_commit" "$tokenizer_remote_commit"
test "$converted_remote_commit" = "$CONVERTED_COMMIT"
test "$tokenizer_remote_commit" = "$TOKENIZER_COMMIT"

printf '%s\n' '=== exact selective ModelScope downloads ==='
download_epoch_start=$(date +%s)
timeout --signal=INT --kill-after=120s 21600s "$PYTHON" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from modelscope import snapshot_download

base = Path(os.environ["DIFFSYNTH_MODEL_BASE_PATH"])
specs = [
    (
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        "master",
        [
            "models_t5_umt5-xxl-enc-bf16.safetensors",
            "Wan2.2_VAE.safetensors",
        ],
    ),
    (
        "Wan-AI/Wan2.1-T2V-1.3B",
        "master",
        [
            "google/umt5-xxl/special_tokens_map.json",
            "google/umt5-xxl/spiece.model",
            "google/umt5-xxl/tokenizer.json",
            "google/umt5-xxl/tokenizer_config.json",
        ],
    ),
]

for repo_id, revision, files in specs:
    root = base / repo_id
    already_complete = [name for name in files if (root / name).is_file()]
    if len(already_complete) == len(files):
        print(f"modelscope_skip_complete repo={repo_id} revision={revision}")
        continue
    result = snapshot_download(
        repo_id,
        revision=revision,
        local_dir=str(root),
        allow_file_pattern=files,
        ignore_file_pattern=already_complete,
        local_files_only=False,
    )
    print(f"modelscope_snapshot repo={repo_id} revision={revision} path={result}")
PY
download_epoch_end=$(date +%s)
printf 'download_wall_seconds=%s\n' "$((download_epoch_end - download_epoch_start))"

printf '%s\n' '=== after: exact payload acceptance ==='
check_existing_or_absent "$CONVERTED_ROOT/$T5_REL" "$T5_BYTES" "$T5_SHA256"
check_existing_or_absent "$CONVERTED_ROOT/$VAE_REL" "$VAE_BYTES" "$VAE_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$SPECIAL_REL" "$SPECIAL_BYTES" "$SPECIAL_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$SPIECE_REL" "$SPIECE_BYTES" "$SPIECE_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$TOKENIZER_JSON_REL" "$TOKENIZER_JSON_BYTES" "$TOKENIZER_JSON_SHA256"
check_existing_or_absent "$TOKENIZER_ROOT/$TOKENIZER_CONFIG_REL" "$TOKENIZER_CONFIG_BYTES" "$TOKENIZER_CONFIG_SHA256"
actual_payload_bytes=$( \
  stat -c '%s' -- \
    "$CONVERTED_ROOT/$T5_REL" \
    "$CONVERTED_ROOT/$VAE_REL" \
    "$TOKENIZER_ROOT/$SPECIAL_REL" \
    "$TOKENIZER_ROOT/$SPIECE_REL" \
    "$TOKENIZER_ROOT/$TOKENIZER_JSON_REL" \
    "$TOKENIZER_ROOT/$TOKENIZER_CONFIG_REL" \
  | awk '{sum += $1} END {print sum}' \
)
test "$actual_payload_bytes" = "$EXPECTED_PAYLOAD_BYTES"
sha256sum -- \
  "$CONVERTED_ROOT/$T5_REL" \
  "$CONVERTED_ROOT/$VAE_REL" \
  "$TOKENIZER_ROOT/$SPECIAL_REL" \
  "$TOKENIZER_ROOT/$SPIECE_REL" \
  "$TOKENIZER_ROOT/$TOKENIZER_JSON_REL" \
  "$TOKENIZER_ROOT/$TOKENIZER_CONFIG_REL"
stat -c 'bytes=%s path=%n' -- \
  "$CONVERTED_ROOT/$T5_REL" \
  "$CONVERTED_ROOT/$VAE_REL" \
  "$TOKENIZER_ROOT/$SPECIAL_REL" \
  "$TOKENIZER_ROOT/$SPIECE_REL" \
  "$TOKENIZER_ROOT/$TOKENIZER_JSON_REL" \
  "$TOKENIZER_ROOT/$TOKENIZER_CONFIG_REL"
dit_file=$(find "$BASE" -type f -name 'diffusion_pytorch_model*.safetensors' -print -quit)
if [[ -n "$dit_file" ]]; then
  printf 'UNEXPECTED_PRETRAINED_DIT=%s\n' "$dit_file" >&2
  exit 53
fi
base_bytes_after=$(tree_bytes "$BASE")
data_free_after=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
home_free_after=$(df -B1 --output=avail /home | tail -n 1 | tr -d '[:space:]')
printf 'actual_payload_bytes=%s\nbase_bytes_after=%s\nbase_tree_delta_bytes=%s\n' \
  "$actual_payload_bytes" "$base_bytes_after" "$((base_bytes_after - base_bytes_before))"
printf 'data_free_bytes_after=%s\nhome_free_bytes_after=%s\n' \
  "$data_free_after" "$home_free_after"
printf 'pretrained_5b_dit_files=0\ntimestamp_end=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' \
  'resume_contract=rerun this unchanged command; same revisions and local-dirs retain ModelScope partial/cache state' \
  'quota_after_required=run shenzhen_mihomo_quota_sanitized_20260821.sh now' \
  'FASTWAM_FW_SZ_310_MODELSCOPE_OK'
