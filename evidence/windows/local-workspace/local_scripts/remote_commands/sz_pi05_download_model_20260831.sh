#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
REV=fa8df6ed103db0f5549c122f3a17c00ba6426c98
MODEL_ROOT=/data/chenyiteng/models/rlinf
PARTIAL="$MODEL_ROOT/.partial-RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed-20260831"
FINAL="$MODEL_ROOT/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed"
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05/bootstrap-20260831
LOG="$RUN/model_download.log"
PIDFILE="$RUN/model_download.pid"

test -x "$VENV/bin/hf"
test ! -e "$FINAL"
test ! -s "$PIDFILE" || ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null
mkdir -p "$MODEL_ROOT" "$PARTIAL" "$RUN"
test "$(df -B1 --output=avail /data | tail -n1)" -gt 12884901888

setsid env VENV="$VENV" REV="$REV" PARTIAL="$PARTIAL" FINAL="$FINAL" bash -c '
  set -euo pipefail
  unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60
  printf "start=%s\n" "$(date --iso-8601=seconds)"
  "$VENV/bin/hf" download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
    --revision "$REV" --local-dir "$PARTIAL" --max-workers 1
  test -s "$PARTIAL/model-00001-of-00003.safetensors"
  test -s "$PARTIAL/model-00002-of-00003.safetensors"
  test -s "$PARTIAL/model-00003-of-00003.safetensors"
  test -s "$PARTIAL/model.safetensors.index.json"
  test -s "$PARTIAL/physical-intelligence/robotwin/norm_stats.json"
  mv "$PARTIAL" "$FINAL"
  printf "end=%s\nmodel=%s\n" "$(date --iso-8601=seconds)" "$FINAL"
  du -sh "$FINAL"
  printf "PI05_MODEL_DOWNLOAD_OK\n"
' > "$LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PIDFILE"
printf 'pid=%s\nlog=%s\nfinal=%s\nPI05_MODEL_DOWNLOAD_STARTED\n' "$pid" "$LOG" "$FINAL"
