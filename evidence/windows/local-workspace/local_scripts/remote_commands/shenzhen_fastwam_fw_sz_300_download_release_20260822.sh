#!/usr/bin/env bash
set -euo pipefail

# FW-SZ-300: official Fast-WAM RoboTwin release, locked to one HF revision.
# Quota is root-only. Run shenzhen_mihomo_quota_sanitized_20260821.sh
# immediately before and after this command and retain all three outputs together.

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
FW_PIN=7faa71108368fbb3b6885649f112af607427a2d4
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
REPO_ID=yuanty/fastwam
REVISION=8eaceeb24c3cc92ff2a9c9a9d266a4941b836705
TARGET=/data/chenyiteng/models/fastwam/release-8eaceeb
CHECKPOINT=robotwin_uncond_3cam_384.pt
STATS=robotwin_uncond_3cam_384_dataset_stats.json
CHECKPOINT_BYTES=12041813092
CHECKPOINT_SHA256=776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63
STATS_BYTES=88715
STATS_SHA256=7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095
EXPECTED_PAYLOAD_BYTES=12041901807
DISK_MARGIN_BYTES=2147483648
HF_CLI="$ENV/bin/huggingface-cli"
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
      exit 40
    fi
    local actual_bytes actual_sha
    actual_bytes=$(stat -c '%s' -- "$path")
    actual_sha=$(sha256sum -- "$path" | awk '{print $1}')
    if [[ "$actual_bytes" != "$expected_bytes" || "$actual_sha" != "$expected_sha" ]]; then
      printf 'REFUSE_MISMATCHED_EXISTING_FILE=%s bytes=%s sha256=%s\n' \
        "$path" "$actual_bytes" "$actual_sha" >&2
      exit 41
    fi
    printf 'existing_complete=%s bytes=%s sha256=%s\n' \
      "$path" "$actual_bytes" "$actual_sha"
  else
    printf 'existing_absent=%s\n' "$path"
  fi
}

printf 'fw_stage=FW-SZ-300\ntimestamp_start=%s\n' "$(date --iso-8601=seconds)"
printf 'repo=%s\nrevision=%s\ntarget=%s\nexpected_payload_bytes=%s\n' \
  "$REPO_ID" "$REVISION" "$TARGET" "$EXPECTED_PAYLOAD_BYTES"
printf '%s\n' \
  'quota_before_contract=run sanitized root-only quota probe immediately before this command' \
  'quota_after_contract=run sanitized root-only quota probe immediately after this command'

test -x "$PYTHON"
test -x "$HF_CLI"
test "$(git -C "$FW" rev-parse HEAD)" = "$FW_PIN"
"$PYTHON" - <<'PY'
from importlib.metadata import version

assert version("huggingface-hub") == "0.29.2", version("huggingface-hub")
print("huggingface_hub=" + version("huggingface-hub"))
PY

printf '%s\n' '=== before: target, capacity, and process-only route ==='
target_bytes_before=$(tree_bytes "$TARGET")
data_free_before=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
home_free_before=$(df -B1 --output=avail /home | tail -n 1 | tr -d '[:space:]')
printf 'target_bytes_before=%s\ndata_free_bytes_before=%s\nhome_free_bytes_before=%s\n' \
  "$target_bytes_before" "$data_free_before" "$home_free_before"
check_existing_or_absent "$TARGET/$CHECKPOINT" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256"
check_existing_or_absent "$TARGET/$STATS" "$STATS_BYTES" "$STATS_SHA256"

missing_bytes=0
[[ -f "$TARGET/$CHECKPOINT" ]] || missing_bytes=$((missing_bytes + CHECKPOINT_BYTES))
[[ -f "$TARGET/$STATS" ]] || missing_bytes=$((missing_bytes + STATS_BYTES))
required_data_free=$((missing_bytes + DISK_MARGIN_BYTES))
printf 'missing_payload_bytes=%s\nrequired_data_free_with_2gib_margin=%s\n' \
  "$missing_bytes" "$required_data_free"
if (( data_free_before < required_data_free )); then
  printf 'INSUFFICIENT_DATA_FREE=%s required=%s\n' "$data_free_before" "$required_data_free" >&2
  exit 42
fi

source /etc/profile.d/mihomo-proxy.sh
mkdir -p \
  "$TARGET" \
  "$CACHE_ROOT/huggingface/hub" \
  "$CACHE_ROOT/tmp"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export TMPDIR="$CACHE_ROOT/tmp"
printf 'HF_HOME=%s\nHF_HUB_CACHE=%s\nHF_ENDPOINT=%s\nTMPDIR=%s\n' \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_ENDPOINT" "$TMPDIR"
env | grep -iE '^(http|https|all|no)_proxy=' \
  | sed -E 's#(https?://)[^/@]+@#\1REDACTED@#I' \
  | sort
curl -sSIL --max-time 30 --connect-timeout 10 -o /dev/null \
  -w 'hf_revision_probe http=%{http_code} remote=%{remote_ip} total=%{time_total}\n' \
  "https://huggingface.co/api/models/$REPO_ID/revision/$REVISION"

printf '%s\n' '=== exact official selective download ==='
download_epoch_start=$(date +%s)
timeout --signal=INT --kill-after=120s 21600s \
  "$HF_CLI" download "$REPO_ID" \
    "$CHECKPOINT" \
    "$STATS" \
    --revision "$REVISION" \
    --local-dir "$TARGET"
download_epoch_end=$(date +%s)
printf 'download_wall_seconds=%s\n' "$((download_epoch_end - download_epoch_start))"

printf '%s\n' '=== after: exact payload acceptance ==='
check_existing_or_absent "$TARGET/$CHECKPOINT" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256"
check_existing_or_absent "$TARGET/$STATS" "$STATS_BYTES" "$STATS_SHA256"
actual_payload_bytes=$( \
  stat -c '%s' -- "$TARGET/$CHECKPOINT" "$TARGET/$STATS" \
  | awk '{sum += $1} END {print sum}' \
)
test "$actual_payload_bytes" = "$EXPECTED_PAYLOAD_BYTES"
sha256sum -- "$TARGET/$CHECKPOINT" "$TARGET/$STATS"
stat -c 'bytes=%s path=%n' -- "$TARGET/$CHECKPOINT" "$TARGET/$STATS"
target_bytes_after=$(tree_bytes "$TARGET")
data_free_after=$(df -B1 --output=avail /data | tail -n 1 | tr -d '[:space:]')
home_free_after=$(df -B1 --output=avail /home | tail -n 1 | tr -d '[:space:]')
printf 'actual_payload_bytes=%s\ntarget_bytes_after=%s\ntarget_tree_delta_bytes=%s\n' \
  "$actual_payload_bytes" "$target_bytes_after" "$((target_bytes_after - target_bytes_before))"
printf 'data_free_bytes_after=%s\nhome_free_bytes_after=%s\n' \
  "$data_free_after" "$home_free_after"
printf 'timestamp_end=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' \
  'resume_contract=rerun this unchanged command; same revision and local-dir retain HF partial/cache state' \
  'quota_after_required=run shenzhen_mihomo_quota_sanitized_20260821.sh now' \
  'FASTWAM_FW_SZ_300_RELEASE_OK'
