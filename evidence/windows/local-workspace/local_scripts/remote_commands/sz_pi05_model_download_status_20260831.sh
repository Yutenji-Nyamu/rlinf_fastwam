#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05/bootstrap-20260831
PARTIAL=/data/chenyiteng/models/rlinf/.partial-RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed-20260831
FINAL=/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed
pid=$(cat "$RUN/model_download.pid")
printf 'pid=%s alive=%s\n' "$pid" "$(kill -0 "$pid" 2>/dev/null && echo yes || echo no)"
if test -d "$PARTIAL"; then du -sh "$PARTIAL"; find "$PARTIAL" -type f -printf '%s %p\n' | sort -nr | head -n 8; fi
if test -d "$FINAL"; then du -sh "$FINAL"; fi
tail -n 30 "$RUN/model_download.log" 2>/dev/null || true
