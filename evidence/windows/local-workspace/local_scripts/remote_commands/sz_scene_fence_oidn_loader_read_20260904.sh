#!/usr/bin/env bash
set -euo pipefail
S=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien
cat "$S/_oidn_tricks.py"
ls -l "$S/oidn_library"
for p in "$S"/oidn_library/*.so*; do
  printf '\n%s\n' "$p"
  readelf -d "$p" | grep -E 'NEEDED|SONAME|RPATH|RUNPATH' || true
done
