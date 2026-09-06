#!/usr/bin/env bash
set -euo pipefail

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
ps -u chenyiteng -o pid=,ppid=,etimes=,stat=,args= --forest | \
  grep -E 'fastwam-7faa-py310-cu128|pip install|conda create|bash -s' | \
  grep -v grep || true
du -sh \
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128 \
  /home/chenyiteng/cache/fastwam-7faa 2>/dev/null || true
