#!/usr/bin/env bash
set -u

URL=https://huggingface.co/api/models/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/revision/92684e50dca1a5f75adc8d332046c4cf4fa7a3d0

probe() {
  local label="$1"
  shift
  printf '%s\n' "=== $label ==="
  curl -4 -L -o /dev/null -sS --connect-timeout 5 --max-time 20 \
    -w 'code=%{http_code} remote=%{remote_ip} connect=%{time_connect} tls=%{time_appconnect} total=%{time_total} bytes=%{size_download}\n' \
    "$@" "$URL"
  printf 'exit=%s\n' "$?"
}

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  bash -c 'curl -4 -L -o /dev/null -sS --noproxy "*" --connect-timeout 5 --max-time 20 -w "code=%{http_code} remote=%{remote_ip} connect=%{time_connect} tls=%{time_appconnect} total=%{time_total} bytes=%{size_download} exit_marker\\n" "$1"; printf "exit=%s\\n" "$?"' _ "$URL"

probe explicit_mihomo --proxy http://127.0.0.1:7890
