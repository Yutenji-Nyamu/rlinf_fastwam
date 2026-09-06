#!/usr/bin/env bash
set -u

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
printf '=== NON_PYC_VENV_FILES_1150_1205 ===\n'
find "$VENV" -xdev -type f -newermt '2026-08-23 11:50:00 UTC' ! -newermt '2026-08-23 12:05:00 UTC' \
  ! -path '*/__pycache__/*' ! -name '*.pyc' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort

FILE="$VENV/lib/python3.11/site-packages/numpydantic/ndarray.pyi"
printf '=== NUMPYDANTIC_STUB_INTEGRITY ===\n'
if test -f "$FILE"; then
  stat -c 'mtime=%y ctime=%z size=%s inode=%i mode=%a' "$FILE"
  sha256sum "$FILE"
  DIST=$(find "$VENV/lib/python3.11/site-packages" -maxdepth 1 -type d -name 'numpydantic-*.dist-info' | head -n1)
  printf 'dist_info=%s\n' "$DIST"
  grep -F 'numpydantic/ndarray.pyi,' "$DIST/RECORD" 2>/dev/null || true
fi

printf 'SZ_VENV_OVERLAP_NARROW_OK\n'
